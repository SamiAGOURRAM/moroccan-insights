import streamlit as st
from langchain_groq import ChatGroq
from src.utils.config import MODEL_CONFIG, INDEX_PATH
from src.utils.database import initialize_database, get_sql_database
from src.models.index_builder import OptimizedIndexBuilder
from src.models.query_generator import EnhancedSQLQueryGenerator
from src.components.chat_interface import ChatInterface

# Initialize session state for components
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_components():
    """Initialize all components (runs only once)"""
    if not st.session_state.initialized:
        try:
            with st.spinner("🚀 Initializing components..."):
                # Initialize LLM
                st.session_state.llm = ChatGroq(**MODEL_CONFIG)
                
                # Initialize database and get dataframe
                st.session_state.df = initialize_database()
                
                # Initialize index builder
                st.session_state.index_builder = OptimizedIndexBuilder(
                    st.session_state.df, 
                    index_path=str(INDEX_PATH)
                )
                
                # Initialize SQL database connection and query generator
                st.session_state.db = get_sql_database()
                st.session_state.query_generator = EnhancedSQLQueryGenerator(
                    db=st.session_state.db,
                    llm=st.session_state.llm
                )
                
                st.session_state.initialized = True
                st.success("✅ All components initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Error initializing components: {str(e)}")
            st.stop()

def main():
    # Page configuration
    st.set_page_config(
        page_title="Morocco Economic Insights",
        page_icon="🇲🇦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
            .main {
                background-color: #f7f7f8;
            }
            .stButton button {
                background-color: #2E86C1;
                color: white;
                border-radius: 0.5rem;
                border: none;
            }
            .stSpinner > div {
                border-top-color: #2E86C1 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize components
    initialize_components()

    # Create chat interface
    chat_interface = ChatInterface(
        llm=st.session_state.llm,
        index_builder=st.session_state.index_builder,
        query_generator=st.session_state.query_generator
    )
    
    # Run chat interface
    chat_interface.run()

if __name__ == "__main__":
    main()