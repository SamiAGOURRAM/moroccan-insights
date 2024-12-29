# Moroccan Economic Insights: Advanced RAG System

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Installation Guide](#installation-guide)
  - [Streamlit Application](#streamlit-application)
  - [Local Development](#local-development)
- [System Components](#system-components)
  - [Data Pipeline](#data-pipeline)
  - [Semantic Search](#semantic-search)
  - [Query Generation](#query-generation)
  - [Visualization Engine](#visualization-engine)
- [Technical Stack](#technical-stack)
- [Future Development](#future-development)
- [Contributing](#contributing)

## Overview

The **Moroccan Economic Insights** platform is an advanced Retrieval-Augmented Generation (RAG) system designed to provide comprehensive, data-driven economic analysis. By integrating semantic search, large language models (LLMs), SQL query generation, and data visualization, this system transforms raw economic indicators into actionable insights.

🔗 **Live Demo**: [Moroccan Insights Platform](https://moroccan-insights-cs.streamlit.app/)

## Key Features

* **Intelligent Data Retrieval**: Semantic search powered by HNSW indexing
* **Multilingual Processing**: Automatic translation and standardization of French-English content
* **Dynamic SQL Generation**: Adaptive query generation using LangChain's SQL Agent
* **Interactive Visualizations**: Rich, interactive plots using Plotly
* **Parallel Processing**: Efficient handling of multiple indicators simultaneously

## Technical Architecture

Our system employs a sophisticated multi-stage pipeline:

1. **Data Preprocessing**: Cleaning, translation, and standardization
2. **Semantic Indexing**: HNSW-based embedding search
3. **Query Processing**: Context-aware indicator extraction
4. **SQL Generation**: Dynamic query construction
5. **Analysis**: Parallel processing of indicators
6. **Visualization**: Interactive data presentation

## Installation Guide

### Streamlit Application

```bash
# Clone the repository
git clone https://github.com/yourusername/moroccan-insights.git
cd moroccan-insights/morocco-economic-insights

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GROQ_API_KEY=your_api_key_here" > .env

# Launch the application
streamlit run app.py
```

### Local Development

For backend testing:

1. Navigate to the `notebooks` directory
2. Ensure the CSV file in `data` directory is present in your environment
3. Follow these steps:

```python

# Initialize components
df = pd.read_csv("path_to_the_csv_file.csv")
db = create_engine("sqlite:///indicators.db")
query_generator = EnhancedSQLQueryGenerator(db=db, llm=llm)
index_builder = OptimizedIndexBuilder(df, index_path='indicators_index')


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

## System Components

### Data Pipeline
- World Bank data preprocessing
- Multilingual text harmonization
- LLM-based summarization
- Data cleaning and standardization

### Semantic Search
- Model: `all-mpnet-base-v2` embeddings
- HNSW indexing for efficient retrieval
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Dynamic similarity thresholds

### Query Generation
- LangChain SQL Agent integration
- Robust error handling
- Fallback query mechanisms
- Context-aware query construction

### Visualization Engine
- Interactive Plotly visualizations
- Time series analysis
- Trend detection
- Parallel processing capabilities

## Technical Stack

| Component | Technology |
|-----------|------------|
| Core Language | Python |
| Embedding Model | `all-mpnet-base-v2` |
| Cross-Encoder | `ms-marco-MiniLM-L-6-v2` |
| LLM | Groq Llama 70B |
| Key Libraries | Sentence Transformers, HNSWLib, Plotly, Pandas, SQLAlchemy, LangChain |
| Database | SQLite |
| Frontend | Streamlit |


## Future Development

1. Multi-country analysis support
2. Advanced trend prediction models
3. Enhanced visualization techniques
4. Expanded multilingual capabilities
5. Integration with online knowledge bases
6. Real-time data updates
7. API endpoint development

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

For bug reports or feature requests, please open an issue in the repository.

---

📊 **Project Status**: Active Development  
📝 **License**: MIT  
👥 **Contributors**: [Sami Agourram]
