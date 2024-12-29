# Moroccan Economic Insights: Advanced RAG System  

## Project Overview  

The **Moroccan Economic Insights** platform is an advanced Retrieval-Augmented Generation (RAG) system designed to deliver actionable insights into Morocco's economy. By leveraging cutting-edge natural language processing, machine learning techniques, and semantic search, this tool transforms raw economic data into meaningful, interactive, and data-driven analyses.  

You can use this RAG solution, available now on this link : https://moroccan-insights-cs.streamlit.app/

### Key Features  

1. **Intelligent Data Retrieval**: Precise searches for contextually relevant economic indicators.  
2. **Semantic Understanding**: Deep contextual analysis of economic metrics using state-of-the-art models.  
3. **Adaptive Analysis**: Dynamic insights tailored to user-specific inquiries.  
4. **Interactive Visualizations**: Data trends visualized for better comprehension and decision-making.  
5. **User Accessibility**: Simplifies complex economic data for diverse user profiles, including researchers, policymakers, and investors.  

---

## Technical Architecture  

The platform employs a sophisticated multi-stage pipeline to process raw data and generate comprehensive insights.  

### Workflow Stages:  

1. **Data Preprocessing**  
2. **Semantic Search**  
3. **Indicator Extraction**  
4. **Data Retrieval**  
5. **Visualization**  
6. **Insight Generation**  

---

## Deep Dive into Key Components  

### 1. Data Preparation  

- **Source**: Economic data is fetched via the World Bank API (CSV for Morocco).  
- **Preprocessing Steps**:  
  - Translation of mixed-language texts (French to English).  
  - Text summarization using a large language model (LLM).  
  - Cleaning and filtering to ensure data relevance.  
  - Standardization of economic indicator descriptions.  

#### Challenges Addressed  
- Multilingual inconsistencies.  
- Preservation of technical terminology.  
- Noise reduction for better data usability.  

---

### 2. Semantic Search Engine  

#### **Indexing with HNSW**  
- **Model**: `all-mpnet-base-v2` for high-quality embeddings.  
- **Algorithm**: Hierarchical Navigable Small World (HNSW) Index for logarithmic search complexity and low computational cost.  

#### **Two-Stage Retrieval**  
1. **HNSW Nearest Neighbor Search**: Fast and scalable retrieval.  
2. **Cross-Encoder Re-Ranking**: Contextual re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` to enhance result precision.  

#### Unique Capabilities  
- Dynamic indicator extraction.  
- Metadata-enhanced queries.  
- Flexible similarity thresholds for tailored searches.  

---

### 3. Adaptive SQL Query Generator  

- **Architecture**: Built with LangChain's SQL Agent for dynamic query generation.  
- **Capabilities**:  
  - Generates SQL queries based on extracted indicators and user inputs.  
  - Handles complex queries with robust error fallback mechanisms.  
  - Ensures flexibility in year ranges and metadata-rich query construction.  

---

### 4. Visualization Strategy  

- **Framework**: Interactive and aesthetically pleasing visualizations built using Plotly.  
- **Key Features**:  
  - Time series plots with trend lines and year-over-year changes.  
  - Dynamic annotations for enhanced insights.  
  - Parallel processing for efficient rendering of multiple indicators.  

---

## Use Cases  

- Economic research and analysis.  
- Investment and market trend identification.  
- Policy planning and decision-making.  

---

## Technical Stack  

| **Aspect**         | **Details**                                            |  
|---------------------|--------------------------------------------------------|  
| **Languages**       | Python                                                 |  
| **Libraries**       | Sentence Transformers, HNSWLib, Plotly, Pandas, SQLAlchemy, LangChain |  
| **Embedding Model** | `all-mpnet-base-v2`                                    |  
| **Cross-Encoder**   | `ms-marco-MiniLM-L-6-v2`                               |  
| **LLM**             | Groq Llama 70B (fast inference)                        |  

---

## Usage Example  

You can either use the available website : https://moroccan-insights-cs.streamlit.app/
or the notebook in the repo.

```python

    # Initialize components
    db = SQLDatabase.from_uri("sqlite:////content/indicators.db")
    query_generator = EnhancedSQLQueryGenerator(db=db, llm=llm)

    question = "What are the top 3 indicators showing Morocco's technological advancement"

    result = analyze_morocco_economy(
        question=question,
        index_builder=index_builder,
        query_generator=query_generator,
        llm=llm
    )

    display_analysis_results(result)
    # Display all visualizations
    for viz_name, fig in result['visualizations'].items():
        print(f"\n{viz_name.upper()}:")
        fig.show()
```  

---

## Future Roadmap  

1. Expand support for multi-country analyses.  
2. Integrate advanced trend prediction models.  
3. Improve visualization techniques for deeper insights.  
4. Increase multilingual support for better global applicability.
5. Use available articles online to enhance the knowledge base

---

## Contributing  

Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.  

---
