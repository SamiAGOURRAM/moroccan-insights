import os
import pickle
from typing import Optional, List, Tuple, Dict, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from hnswlib import Index
import numpy as np
from langchain.prompts import PromptTemplate
import pandas as pd

class OptimizedIndexBuilder:
    def __init__(self, df, index_path: Optional[str] = None):
        """
        Initialize the search engine with highly optimized parameters and re-ranking capability

        Args:
            df: DataFrame containing indicators and descriptions
            index_path: Path to save/load the index and related data
        """
        # Initialize with a more powerful model for better semantic understanding
        self.model = SentenceTransformer('all-mpnet-base-v2')

        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Clean and prepare data with enhanced text combination
        self.descriptions = df['indicator_description'].dropna().tolist()
        self.indicators = df[df['indicator_description'].notna()]['indicator_name'].tolist()

        # Store original dataframe for later reference
        self.df = df

        # Combine texts with more emphasis on indicator names
        self.combined_texts = [
            f"Indicator: {ind.strip()} Description: {desc.strip()}"
            for ind, desc in zip(self.indicators, self.descriptions)
        ]

        # Get dimension from model
        self.dim = self.model.get_sentence_embedding_dimension()

        # Optimized parameters for small dataset
        self.ef_construction = 400
        self.M = 128
        self.ef_search = 200

        # Initialize prompt template for LLM
        self.description_prompt = PromptTemplate.from_template("""
          Question: {question}

          You are a Economics specialist. Follow these rules strictly:
          1. Based on the question, Generate AT MOST 2 World Bank indicator descriptions relevant to the context
          2. Provide a clear, professional description in English that explains:
            - What this indicator measures
            - Its importance
            - How it's typically used
            - Keep it concise but informative. Less than 50 words

          3. Focus only on the most relevant economic indicators

          Format each indicator as:
          [Indicator Name], [Abbreviation]: [Description]

          Example:
          Goods imports by the reporting country, IMPS: This indicator measures the total value of goods imported by a country from the rest of the world, in current US dollars. It's essential for tracking trade balances, economic growth, and competitiveness. Typically used by policymakers, researchers, and businesses to analyze trade patterns, identify opportunities, and inform investment decisions.
          """)

        if index_path and self._try_load_index(index_path):
            print(f"Successfully loaded index from {index_path}")
        else:
            print("Building new index with optimized parameters...")
            self._build_index()
            if index_path:
                self._save_index(index_path)
                print(f"Saved optimized index to {index_path}")

    def _build_index(self):
        """Build the HNSW index with optimized parameters for maximum accuracy"""
        self.index = Index(space='cosine', dim=self.dim)

        self.index.init_index(
            max_elements=len(self.combined_texts),
            ef_construction=self.ef_construction,
            M=self.M
        )

        self.index.set_ef(self.ef_search)

        print("Generating embeddings...")
        embeddings = self.model.encode(
            self.combined_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.index.add_items(embeddings)
    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search for most similar indicators given a query

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of tuples (indicator_name, description, similarity_score)
        """
        # Enhance query with prefix for better matching
        enhanced_query = f"Indicator description: {query.strip()}"

        # Generate query embedding
        query_vector = self.model.encode(
            [enhanced_query],
            normalize_embeddings=True
        )

        # Search with higher ef value for better accuracy
        self.index.set_ef(max(self.ef_search, k * 2))  # Dynamically adjust ef based on k

        # Get nearest neighbors
        labels, distances = self.index.knn_query(query_vector, k=k)

        # Convert distances to similarities (cosine distance to similarity)
        similarities = 1 - distances[0]

        # Return results with scores
        results = []
        for idx, sim in zip(labels[0], similarities):
            results.append((
                self.indicators[idx],
                self.descriptions[idx],
                float(sim)
            ))

        return results

    def search_with_reranking(self,
                            query: str,
                            k: int = 5,
                            initial_k: int = 20) -> List[Dict[str, Union[str, float]]]:
        """
        Search for most similar indicators with re-ranking

        Args:
            query: Search query text
            k: Number of final results to return
            initial_k: Number of initial candidates for re-ranking

        Returns:
            List of dictionaries containing matched indicators with scores
        """
        # 1. Initial search with larger k
        initial_results = self.search(query, k=initial_k)

        # 2. Prepare pairs for re-ranking
        pairs = []
        for indicator, description, _ in initial_results:
            # Combine indicator and description for better context
            full_text = f"Indicator: {indicator} Description: {description}"
            pairs.append((query, full_text))

        # 3. Re-rank using cross-encoder
        cross_scores = self.cross_encoder.predict(pairs)

        # 4. Combine results with new scores
        reranked_results = []
        for idx, (indicator, description, bi_score) in enumerate(initial_results):
            reranked_results.append({
                'indicator_name': indicator,
                'description': description,
                'bi_encoder_score': float(bi_score),
                'cross_encoder_score': float(cross_scores[idx]),
                'final_score': float(cross_scores[idx])  # Use cross-encoder score as final
            })

        # 5. Sort by cross-encoder score and take top k
        reranked_results = sorted(
            reranked_results,
            key=lambda x: x['final_score'],
            reverse=True
        )[:k]

        return reranked_results

    def enhanced_search(
                self,
                llm,
                question: str,
                k: int = 5,
                initial_k: int = 20,
                similarity_threshold: float = 0.5,
                min_k: int = 3,
                max_k: int = 15
            ) -> Dict[str, List]:
        """
        Enhanced search using LLM-generated descriptions and re-ranking

        Args:
            llm: Language model instance
            question: User question to find relevant indicators
            k: Number of final results to return
            initial_k: Number of initial candidates for re-ranking
            similarity_threshold: Minimum similarity score to include

        Returns:
            Dictionary with indicator names as keys and [description, confidence] as values
        """
        try:
            extracted_k = self._extract_k_from_question(llm, question)
            if extracted_k is not None:
                k = max(min(extracted_k, max_k), min_k)
                initial_k = max(k * 3, initial_k)

            print(f"Searching for {k} indicators...")
            # 1. Generate descriptions using LLM
            llm_response = llm.predict(self.description_prompt.format(question=question))
            parsed_indicators = self._parse_llm_response(llm_response)

            if not parsed_indicators:
                return {}

            # 2. Search and re-rank for each LLM-generated description
            all_results = []

            for indicator in parsed_indicators:
                enhanced_query = f"Context: {question} Description: {indicator['description']}"

                # Get re-ranked results for this query
                results = self.search_with_reranking(
                    enhanced_query,
                    k=k,
                    initial_k=initial_k
                )
                all_results.extend(results)

            # 3. Deduplicate and keep highest scores
            final_dict = {}
            for result in all_results:
                indicator_name = result['indicator_name']
                if (indicator_name not in final_dict or
                    result['final_score'] > final_dict[indicator_name][1]):
                    final_dict[indicator_name] = [
                        result['description'],
                        float(result['final_score'])
                    ]

            # 4. Filter by threshold and take top k
            filtered_dict = {
                ind: values
                for ind, values in final_dict.items()
                if values[1] >= similarity_threshold
            }

            # Sort by confidence and limit to top k
            sorted_items = sorted(
                filtered_dict.items(),
                key=lambda x: x[1][1],
                reverse=True
            )[:k]

            return dict(sorted_items)

        except Exception as e:
            print(f"Error during enhanced search: {str(e)}")
            return {}


    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response into structured format, handling various response patterns

        Args:
            response: Raw LLM response text

        Returns:
            List of dictionaries containing parsed indicators
        """
        parsed_indicators = []

        try:
            # Remove the template text if present
            response = response.replace('Answer Format : {indicator_name, abbreviation (if exists): description}', '')

            # Split into individual indicators, handling multiple newlines
            indicators = [ind.strip() for ind in response.split('\n') if ind.strip()]

            current_indicator = {}

            for line in indicators:
                # Skip empty lines
                if not line or line.isspace():
                    continue

                # Check if this is a new indicator (contains colon)
                if ':' in line:
                    # Save previous indicator if exists
                    if current_indicator:
                        parsed_indicators.append(current_indicator)
                        current_indicator = {}

                    # Split into name part and description
                    name_part, description = line.split(':', 1)

                    # Handle abbreviation if present
                    if ',' in name_part:
                        name, abbr = name_part.split(',', 1)
                        current_indicator = {
                            'name': name.strip(),
                            'abbreviation': abbr.strip(),
                            'description': description.strip()
                        }
                    else:
                        current_indicator = {
                            'name': name_part.strip(),
                            'description': description.strip()
                        }
                else:
                    # If no colon, append to current description if exists
                    if current_indicator:
                        current_indicator['description'] = current_indicator['description'] + ' ' + line.strip()

            # Add the last indicator if exists
            if current_indicator:
                parsed_indicators.append(current_indicator)

        except Exception as e:
            print(f"Error in parsing LLM response: {str(e)}")
            print(f"Problematic response: {response}")

        return parsed_indicators

    def _save_index(self, path: str):
        """Save the index and related data with additional metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.index.save_index(f"{path}.idx")

        metadata = {
            'descriptions': self.descriptions,
            'indicators': self.indicators,
            'combined_texts': self.combined_texts,
            'dim': self.dim,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'ef_search': self.ef_search
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)

    def _try_load_index(self, path: str) -> bool:
        """Load index with verification of metadata"""
        try:
            if not (os.path.exists(f"{path}.idx") and os.path.exists(f"{path}.meta")):
                return False

            with open(f"{path}.meta", 'rb') as f:
                metadata = pickle.load(f)

            if (len(metadata['descriptions']) != len(self.descriptions) or
                len(metadata['indicators']) != len(self.indicators)):
                print("Warning: Saved index doesn't match current data dimensions")
                return False

            self.index = Index(space='cosine', dim=self.dim)
            self.index.load_index(f"{path}.idx")
            self.index.set_ef(self.ef_search)

            return True

        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False

    def _extract_k_from_question(self, llm, question: str) -> Optional[int]:
        """
        Use LLM to extract the number of indicators requested from the question

        Args:
            llm: Language model instance
            question: User's question

        Returns:
            Optional[int]: Number of indicators requested, None if not specified
        """
        extraction_prompt = """
        Extract the number of indicators requested in this question. If no specific number is mentioned, return "default".
        Only return the number (or "default") without any explanation.

        Examples:
        Question: "What are the top 10 indicators for economic growth?"
        Answer: 10

        Question: "Show me the five most relevant indicators"
        Answer: 5

        Question: "What are thee most important indicators" (typo in 'three')
        Answer: 3

        Question: "Can you give me important indicators for infrastructure?"
        Answer: default

        Current question: {question}
        """

        try:
            # Ask LLM to extract the number
            response = llm.predict(extraction_prompt.format(question=question)).strip().lower()

            # Handle the response
            if response == "default":
                return None

            try:
                # Convert LLM response to integer
                k = int(response)
                return k
            except ValueError:
                print(f"Could not convert LLM response '{response}' to number, using default")
                return None

        except Exception as e:
            print(f"Error in number extraction: {str(e)}")
            return None
