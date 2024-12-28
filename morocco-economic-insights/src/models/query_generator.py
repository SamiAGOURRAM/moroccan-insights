from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType 
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import sqlalchemy

class EnhancedSQLQueryGenerator:
    def __init__(self, db: SQLDatabase, llm):
        """
        Initialize Enhanced SQL Query Generator

        Args:
            db: SQLDatabase instance
            llm: Language model instance
        """
        self.db = db
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Get the underlying SQLAlchemy engine
        self.engine = sqlalchemy.create_engine(str(self.db._engine.url))

        # Initialize SQL agent with specific configuration
        # Initialize SQL agent with specific configuration
        self.agent = create_sql_agent(
            llm=llm,
            toolkit=self.toolkit,
            agent_type="zero-shot-react-description",  # Use string instead of AgentType
            verbose=True
        )
        # Improved prompt template with clear structure and examples
        self.query_template = PromptTemplate.from_template("""
        You are an expert SQL query generator for the World Bank indicators database.
        Generate a SQL query to analyze economic indicators for Morocco.

        Context:
        - Database contains economic indicators for Morocco from 2010-2023
        - Each indicator has yearly values and metadata
        - Question: {question}

        Relevant Indicators to analyze:
        {indicators_details}

        Table Schema:
        Table: moroccan_indicators
        Columns:
        - indicator_name (text): Name of the indicator
        - indicator_description (text): Description of what the indicator measures
        - SOURCE_ORGANIZATION (text): Organization that provided the data
        - [2010-2023]: Separate columns for each year's values

        Requirements:
        1. Query MUST include all specified indicators using WHERE indicator_name IN (...)
        2. Select only years relevant to the question
        3. Include indicator metadata (name, description, source)
        4. Order results logically
        5. Handle NULL values appropriately

        Return ONLY the SQL query without any additional text or explanations.
        """)

    def sanitize_indicator_name(self, name: str) -> str:
        """
        Sanitize the indicator name while carefully handling complex quote formatting.

        Args:
            name: Raw indicator name as a string.

        Returns:
            Sanitized indicator name as a string.
        """
        # Strip leading and trailing whitespace
        name = name.strip()
        
        # Remove multiple consecutive single quotes at the start
        while name.startswith("'") and name.count("'") > 1:
            name = name.lstrip("'")
        
        # Remove multiple consecutive single quotes at the end
        while name.endswith("'") and name.count("'") > 1:
            name = name.rstrip("'")
        
        # Remove surrounding double quotes if present
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        
        # Escape single quotes for SQL safety
        name = name.replace("'", "''")
        
        return name.strip()

    def format_indicators_for_prompt(self, indicators: Dict[str, List[Any]]) -> str:
        formatted_list = []
        for name, (description, score) in indicators.items():
            sanitized_name = self.sanitize_indicator_name(name)
            formatted_list.append(f"- {sanitized_name}\n  Description: {description}\n  Relevance Score: {score:.2f}")
        return "\n".join(formatted_list)



    def generate_query(
        self,
        question: str,
        indicators: Dict[str, List[Any]],
        year_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate SQL query based on question and indicators

        Args:
            question: User's question about Morocco's economy
            indicators: Dictionary of relevant indicators
            year_range: Optional tuple of (start_year, end_year)

        Returns:
            Generated SQL query string
        """
        try:
            
            # Format indicators for prompt
            indicators_details = self.format_indicators_for_prompt(indicators)

            # Generate prompt
            prompt = self.query_template.format(
                question=question,
                indicators_details=indicators_details
            )

            # Get query from agent using invoke instead of run
            response = self.agent.invoke({"input": prompt})
            
            # Handle the response based on its type
            if isinstance(response, dict) and "output" in response:
                response = response["output"]
            elif isinstance(response, str):
                response = response
            else:
                raise ValueError("Unexpected response format from agent")

            # Extract and validate query
            query = self._extract_sql_query(response)

            if not self._validate_query(query, indicators.keys()):
                query = self._create_fallback_query(indicators.keys(), year_range)

            return query

        except Exception as e:
            print(f"Error in query generation: {str(e)}")
            return self._create_fallback_query(indicators.keys(), year_range)

    def _extract_sql_query(self, response: str) -> str:
        """
        Extract clean SQL query from agent response

        Args:
            response: Raw response from agent

        Returns:
            Clean SQL query string
        """
        # Remove markdown code blocks if present
        if '```sql' in response:
            query = response.split('```sql')[1].split('```')[0].strip()
        elif '```' in response:
            query = response.split('```')[1].split('```')[0].strip()
        else:
            # Extract query starting with SELECT
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            query_lines = []
            capture = False

            for line in lines:
                if line.upper().startswith('SELECT'):
                    capture = True
                if capture:
                    query_lines.append(line)
                    if line.strip().endswith(';'):
                        break

            query = ' '.join(query_lines)

        # Clean and standardize query
        query = query.strip()
        if not query.endswith(';'):
            query += ';'

        return query

    def _validate_query(self, query: str, indicator_names: List[str]) -> bool:
        """
        Validate generated query meets requirements

        Args:
            query: SQL query to validate
            indicator_names: List of indicator names that should be included

        Returns:
            Boolean indicating if query is valid
        """
        if not query:
            return False

        query_upper = query.upper()

        # Basic syntax checks
        required_elements = [
            'SELECT',
            'FROM MOROCCAN_INDICATORS',
            'WHERE',
            'IN ('
        ]

        # Check all required elements are present
        for element in required_elements:
            if element.upper() not in query_upper:
                return False

        # More flexible indicator checking
        try:
            # Extract the IN clause and clean it
            in_clause = query_upper.split('IN')[1].split(')')[0].replace('(', '').replace("'", '').strip()
            indicators_in_query = [ind.strip() for ind in in_clause.split(',')]

            # Sanitize both original and query indicators
            sanitized_original = [self.sanitize_indicator_name(ind) for ind in indicator_names]
            
            # Check if all original indicators are represented in some form
            for orig_ind in sanitized_original:
                if not any(orig_ind.upper() in query_ind.upper() for query_ind in indicators_in_query):
                    return False

        except Exception:
            return False

        return True

    def _create_fallback_query(
        self,
        indicator_names: List[str],
        year_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Create safe fallback query when generation fails

        Args:
            indicator_names: List of indicator names to include
            year_range: Optional tuple of (start_year, end_year)

        Returns:
            Safe fallback query string
        """
        # Sanitize indicators for the IN clause
        sanitized_indicators = [self.sanitize_indicator_name(name) for name in indicator_names]

        # Format indicators for IN clause
        indicators_str = "', '".join(sanitized_indicators)

        # Determine year columns
        if year_range:
            start_year, end_year = year_range
        else:
            start_year, end_year = 2010, 2023

        year_columns = [f'"{year}"' for year in range(start_year, end_year + 1)]
        years_sql = ", ".join(year_columns)

        # Construct safe query
        query = f"""
        SELECT
            indicator_name,
            indicator_description,
            SOURCE_ORGANIZATION,
            {years_sql}
        FROM moroccan_indicators
        WHERE indicator_name IN ('{indicators_str}')
        ORDER BY indicator_name;
        """

        return query

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with query results
        """
        try:
            # Use the engine directly for query execution
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)

                # Clean up column names
                df.columns = df.columns.str.strip('"')

                # Convert year columns to numeric
                year_columns = [col for col in df.columns if col.isdigit()]
                for col in year_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                return df

        except Exception as e:
            print(f"Error executing query: {str(e)}")
            print(f"Problematic query: {query}")
            return None
