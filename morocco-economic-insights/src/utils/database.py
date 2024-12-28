import pandas as pd
import sqlite3
from .config import DB_PATH, CSV_PATH
from langchain.sql_database import SQLDatabase

def initialize_database():
    """Initialize database if it doesn't exist"""
    if not DB_PATH.exists():
        print("Database not found. Creating new database...")
        # Read CSV
        df = pd.read_csv(CSV_PATH)
        
        # Create SQLite database
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('moroccan_indicators', conn, if_exists='replace', index=False)
        conn.close()
        print("Database created successfully!")
        return df
    else:
        print("Using existing database...")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM moroccan_indicators", conn)
        conn.close()
        return df

def get_sql_database():
    """Get SQLDatabase instance"""
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")