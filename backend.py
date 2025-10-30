"""
Backend service module for search functionality.
Maintains separation between frontend (Streamlit) and backend (search logic).
"""
from typing import List, Dict, Any, Literal
from search import Search, HSearchResults
from shared_dataclasses import Predicate
from settings import DISPLAY_COLS, PREDICATE_COLUMNS
import pandas as pd


class SearchService:
    """
    Service class that wraps the search functionality and provides
    a clean interface for the frontend.
    """
    
    def __init__(self):
        """Initialize the search service."""
        self.search = Search()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the search service and release resources."""
        if hasattr(self, 'search'):
            self.search.close()
    
    def perform_search(
        self,
        query: str,
        k: int,
        predicates: List[Dict[str, Any]],
        method: Literal["pre_search", "post_search"] = "post_search"
    ) -> pd.DataFrame:
        """
        Perform a search with the given parameters.
        
        Args:
            query: The search query text
            k: Number of results to return
            predicates: List of predicate dictionaries with keys: 'key', 'value', 'operator'
            method: Search method to use ('pre_search' or 'post_search')
        
        Returns:
            DataFrame with search results including similarity scores
        """
        # Convert predicate dictionaries to Predicate objects
        predicate_objects = []
        for pred in predicates:
            try:
                predicate_objects.append(
                    Predicate(
                        key=pred['key'],
                        value=pred['value'],
                        operator=pred['operator']
                    )
                )
            except (ValueError, KeyError) as e:
                # Skip invalid predicates
                print(f"Warning: Skipping invalid predicate {pred}: {e}")
                continue
        
        # Perform search
        results: HSearchResults = self.search.search(
            query=query,
            predicates=predicate_objects,
            k=k,
            method=method
        )
        
        # Convert to DataFrame
        df = results.to_df(show_cols=DISPLAY_COLS)
        
        return df
    
    @staticmethod
    def get_available_predicate_columns() -> List[str]:
        """Get list of columns that can be used as predicates."""
        return PREDICATE_COLUMNS
    
    @staticmethod
    def get_available_operators() -> List[str]:
        """Get list of available operators for predicates."""
        return ["=", ">", "<", ">=", "<=", "IN"]
    
    @staticmethod
    def get_predicate_type(column: str) -> str:
        """
        Get the data type for a predicate column.
        
        Returns:
            'date', 'int', or 'str'
        """
        from settings import DATE_COLUMNS, INT_COLUMNS_DB
        
        if column in DATE_COLUMNS:
            return 'date'
        elif column in INT_COLUMNS_DB:
            return 'int'
        else:
            return 'str'
