import streamlit as st
from typing import Dict, Any
from src.models.analyzer import analyze_morocco_economy
from src.components.visualizations import display_visualizations

class ChatInterface:
    def __init__(self, llm, index_builder, query_generator):
        """
        Initialize ChatInterface component
        
        Args:
            llm: Language model instance
            index_builder: OptimizedIndexBuilder instance
            query_generator: EnhancedSQLQueryGenerator instance
        """
        self.llm = llm
        self.index_builder = index_builder
        self.query_generator = query_generator
        
        # Initialize session state for messages if not exists
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def _display_message(self, message: Dict[str, Any], key: int):
        """Display a single chat message with styling"""
        message_type = message["role"]
        cols = st.columns([1, 10])
        
        # Display avatar and username
        with cols[0]:
            if message_type == "user":
                st.markdown("👤 **You**")
            else:
                st.markdown("🤖 **Assistant**")
        
        # Display message content
        with cols[1]:
            if message_type == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", 
                          unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", 
                          unsafe_allow_html=True)
                
                # Display visualizations if present
                if "visualizations" in message:
                    display_visualizations(message["visualizations"], message_idx=key)

    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the analysis results into a chat message"""
        if "error" in result:
            return {
                "role": "assistant",
                "content": f"❌ Error: {result['error']}"
            }
        
        # Format the insights and indicators into a readable message
        response_content = f"""
        ### 📊 Analysis Results

        {result['insights']}

        ### 📈 Indicators Analyzed:
        """
        
        for idx, indicator in enumerate(result['indicators_details'], 1):
            response_content += f"""
            {idx}. **{indicator['name']}**
               - Source: {indicator['source']}
               - Description: {indicator['description']}
            """
        
        return {
            "role": "assistant",
            "content": response_content,
            "visualizations": result['visualizations']
        }

    def _handle_user_input(self):
        """Handle user input and generate response"""
        # Create columns for input and button
        col1, col2 = st.columns([8, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about Morocco's economy:",
                key="user_input",
                placeholder="e.g., What are the top 3 indicators showing Morocco's technological advancement?"
            )
        
        with col2:
            send_button = st.button("Send", use_container_width=True)
        
        return user_input, send_button

    def _render_sidebar(self):
        """Render sidebar with information and controls"""
        with st.sidebar:
            st.title("Morocco Economic Insights")
            st.markdown("""
            Welcome to the Morocco Economic Insights chatbot! 
            
            Ask questions about Morocco's economy and get detailed insights with visualizations.
            
            **Example questions:**
            - What are the top 3 indicators showing Morocco's technological advancement?
            - How has Morocco's GDP growth evolved in the last 5 years?
            - Show me indicators related to Morocco's education sector
            - What are the key economic indicators for foreign investment?
            - Analyze Morocco's trade balance indicators
            """)
            
            # Add clear chat button
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            # Add about section
            with st.expander("ℹ️ About"):
                st.markdown("""
                A RAG-powered analytics system that helps investors understand Morocco's economic landscape through natural language queries. Leveraging World Bank indicators (2010-2023), it combines advanced retrieval techniques with AI to provide targeted insights and data visualizations for informed decision-making.

                ## Key Features
                - **Intelligent RAG Pipeline**: Combines semantic search, SQL generation, and LLM analysis for accurate, context-aware responses
                - **Natural Language Interface**: Ask questions about Morocco's economy in plain language to get relevant indicators and insights
                - **Dynamic Data Visualization**: Clear, focused time series analysis of economic indicators with trend detection

                ## Technologies
                Built with Python, LangChain, and Sentence Transformers for the RAG implementation, integrated with Plotly for interactive visualizations and SQL for efficient data retrieval.

                ## Supervised By: 
                Pr.Echihabi

                """)

    def run(self):
        """Run the chat interface"""
        # Apply custom CSS
        st.markdown("""
        <style>
            .user-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .assistant-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                
            }
            
            .stButton button {
                background-color: #2E86C1;
                color: white;
                border-radius: 0.5rem;
                border: none;
                padding: 0.5rem 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Render sidebar
        self._render_sidebar()
        
        # Main chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                self._display_message(message, idx)
        
        # Handle user input
        user_input, send_button = self._handle_user_input()
        
        # Process user input and generate response
        if send_button and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Show thinking message
            with st.spinner("🔄 Analyzing Morocco's economic indicators..."):
                try:
                    # Get analysis results
                    result = analyze_morocco_economy(
                        question=user_input,
                        index_builder=self.index_builder,
                        query_generator=self.query_generator,
                        llm=self.llm
                    )
                    
                    # Format and add assistant response
                    assistant_message = self._format_response(result)
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            
            # Rerun to update the UI
            st.rerun()