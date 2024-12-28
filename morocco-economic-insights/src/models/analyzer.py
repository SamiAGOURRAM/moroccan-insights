from .index_builder import OptimizedIndexBuilder
from .query_generator import EnhancedSQLQueryGenerator
from typing import Dict, List, Any
import concurrent.futures

import pandas as pd
import plotly.graph_objects as go
import numpy as np

def create_visualizations(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """Create individual time series visualizations for each indicator"""
    visualizations = {}
    try:
        # Get year columns (only numeric columns)
        year_columns = [col for col in df.columns if str(col).isdigit()]
        
        for indicator in df['indicator_name'].unique():
            # Get indicator data
            ind_data = df[df['indicator_name'] == indicator].copy()
            if ind_data.empty:
                continue

            # Convert year columns to numeric and handle NaN values
            years = []
            values = []
            for col in year_columns:
                val = pd.to_numeric(ind_data[col].iloc[0], errors='coerce')
                if not pd.isna(val):  # Only include non-NaN values
                    years.append(int(col))
                    values.append(float(val))

            source = ind_data['SOURCE_ORGANIZATION'].iloc[0] if 'SOURCE_ORGANIZATION' in ind_data.columns else 'Unknown'

            # Create the data structure directly
            plot_data = {
                'data': [
                    {
                        'x': years,
                        'y': values,
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': 'Actual Values',
                        'line': {'width': 2},
                        'marker': {'size': 8}
                    }
                ],
                'layout': {
                    'title': {
                        'text': f"{indicator}<br><sup>Source: {source}</sup>",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    'xaxis': {'title': 'Year'},
                    'yaxis': {'title': 'Value'},
                    'hovermode': 'x unified',
                    'template': 'plotly_white',
                    'height': 600,
                    'showlegend': True,
                    'legend': {
                        'orientation': 'h',
                        'yanchor': 'bottom',
                        'y': 1.02,
                        'xanchor': 'right',
                        'x': 1
                    }
                }
            }

            # Add trend line if we have enough points
            if len(years) > 1:
                z = np.polyfit(years, values, 1)
                p = np.poly1d(z)
                trend_values = [float(p(year)) for year in years]  # Convert to float to ensure JSON serialization

                plot_data['data'].append({
                    'x': years,
                    'y': trend_values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Trend Line',
                    'line': {'dash': 'dot', 'width': 1}
                })

            visualizations[f'indicator_{indicator}'] = plot_data

        return visualizations

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_morocco_economy(
    question: str,
    index_builder: OptimizedIndexBuilder,
    query_generator: EnhancedSQLQueryGenerator,
    llm,
    default_k: int = 3
) -> dict:
    """Enhanced analysis pipeline with visualizations and optimized output"""
    print(f"\nProcessing question: {question}")

    try:
        # 1. Extract requested number of indicators
        extracted_k = index_builder._extract_k_from_question(llm, question)
        k = extracted_k if extracted_k is not None else default_k

        # 2. Find relevant indicators
        print(f"\nSearching for {k} indicators...")
        relevant_indicators = index_builder.enhanced_search(
            llm=llm,
            question=question,
            k=k,
            initial_k=max(k * 3, 10),
            similarity_threshold=0.6
        )

        if not relevant_indicators:
            return {"error": "No relevant indicators found for the question"}

        # 3. Generate and execute SQL query
        print("\nGenerating and executing SQL query...")
        sql_query = query_generator.generate_query(
            question=question,
            indicators=relevant_indicators
        )

        if not sql_query:
            return {"error": "Failed to generate SQL query"}

        results_df = query_generator.execute_query(sql_query)

        if results_df is None or results_df.empty:
            return {"error": "No data found for the specified indicators"}

        # 4. Create visualizations
        print("\nGenerating visualizations...")
        visualizations = create_visualizations(results_df, question)

        # Enhanced insights prompt
        insights_prompt = f"""
        You are an Economics expert
        Provide a comprehensive objective economic analysis for Morocco based on the following:

        User Question: {question}

        Available Data:
        {results_df.to_string()}

        Provide an analysis that includes:

        1. Executive Summary:
           - Key findings directly addressing the user's question
           - Overall trends and patterns

        2. Detailed Analysis:
           - Individual indicator performance and trends
           - Year-over-year changes and their significance
           - Notable inflection points or shifts

        3. Market Implications:
           - Impact on investment opportunities
           - Potential risks and challenges
           - Comparative advantage aspects

        4. Forward-Looking Insights:
           - Future trajectory based on historical trends
           - Key areas to monitor
           - Strategic recommendations

        Guidelines:
        - Focus on ONLY on objective data-driven insights
        - Highlight both opportunities and risks
        - Provide specific, actionable insights

        Format the response with clear sections and use bullet points for key findings.
        """

        # Get insights
        insights = llm.predict(insights_prompt)

        # 5. Prepare enhanced output
        indicator_details = []
        for ind_name in results_df['indicator_name'].unique():
            source = results_df[results_df['indicator_name'] == ind_name]['SOURCE_ORGANIZATION'].iloc[0] if 'SOURCE_ORGANIZATION' in results_df.columns else 'Unknown'
            description = relevant_indicators[ind_name][0]
            indicator_details.append({
                'name': ind_name,
                'source': source,
                'description': description
            })

        return {
            "question": question,
            "indicators_details": indicator_details,
            "insights": insights,
            "visualizations": visualizations,
            "sql_query": sql_query
        }

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}