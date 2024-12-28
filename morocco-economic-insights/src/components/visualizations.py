import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
import numpy as np

def display_visualizations(visualizations: Dict[str, Any], message_idx: int = 0):
    """
    Display visualization components for the chat interface
    
    Args:
        visualizations: Dictionary containing Plotly figures for each indicator
        message_idx: Index of the current message for unique keys
    """
    if not visualizations:
        return
    
    # Create tabs for different visualization types
    tab_names = ["Time Series"]
    tabs = st.tabs([f"{name}_{message_idx}" for name in tab_names])
    
    with tabs[0]:  # Time Series tab
        st.markdown("### 📈 Time Series Analysis")
        for viz_idx, (viz_name, fig) in enumerate(visualizations.items()):
            with st.expander(f"📊 {viz_name}", expanded=True):
                # Create a unique key for each plotly chart
                unique_key = f"viz_{message_idx}_{viz_idx}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
                
                # Add insights if available
                if hasattr(fig, 'insights'):
                    st.info(fig.insights)

def create_time_series_plot(
    data: Dict[str, Any],
    title: str,
    x_data: list,
    y_data: list,
    source: str = None
) -> go.Figure:
    """
    Create a time series plot with trend analysis
    
    Args:
        data: Dictionary containing plot data
        title: Plot title
        x_data: X-axis data (usually years)
        y_data: Y-axis data (values)
        source: Data source
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        name='Actual Values',
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=8)
    ))
    
    # Add trend line if we have enough data points
    mask = ~np.isnan(y_data)
    if np.sum(mask) > 1:
        z = np.polyfit(np.array(x_data)[mask], np.array(y_data)[mask], 1)
        p = np.poly1d(z)
        trend_values = p(x_data)
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=trend_values,
            name='Trend Line',
            line=dict(dash='dot', width=1)
        ))
        
        # Calculate trend metrics
        slope = z[0]
        trend_direction = "upward" if slope > 0 else "downward"
        avg_change = slope * len(x_data)
        
        # Add trend annotation
        fig.add_annotation(
            text=f"Overall Trend: {trend_direction}<br>Average Change: {avg_change:.2f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    
    # Calculate year-over-year changes
    yoy_changes = np.diff(y_data) / y_data[:-1] * 100
    
    # Add YoY changes as bar chart
    fig.add_trace(go.Bar(
        x=x_data[1:],
        y=yoy_changes,
        name='YoY Change (%)',
        yaxis='y2',
        opacity=0.3
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sup>Source: {source}</sup>" if source else title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Year',
        yaxis_title='Value',
        yaxis2=dict(
            title='YoY Change (%)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_comparison_plot(
    data: Dict[str, list],
    title: str,
    categories: list,
    values: list,
    reference_value: float = None
) -> go.Figure:
    """
    Create a comparison plot (bar chart with optional reference line)
    
    Args:
        data: Dictionary containing plot data
        title: Plot title
        categories: Category names
        values: Category values
        reference_value: Optional reference value to show as line
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add reference line if provided
    if reference_value is not None:
        fig.add_hline(
            y=reference_value,
            line_dash="dash",
            annotation_text="Reference",
            annotation_position="bottom right"
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_tickangle=-45,
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def add_visualization_insights(fig: go.Figure, insights: str) -> go.Figure:
    """
    Add insights to a visualization figure
    
    Args:
        fig: Plotly figure object
        insights: Insights text to add
        
    Returns:
        Updated Plotly figure object
    """
    fig.insights = insights
    return fig