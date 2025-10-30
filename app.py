"""
Minimalist Streamlit Frontend for Hybrid Search
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from backend import SearchService


def initialize_session_state():
    """Initialize session state variables."""
    if 'predicates' not in st.session_state:
        st.session_state.predicates = []


def add_predicate(key: str, operator: str, value):
    """Add a predicate to the session state."""
    if key and value is not None:
        predicate = {
            'key': key,
            'operator': operator,
            'value': value
        }
        st.session_state.predicates.append(predicate)


def remove_predicate(index: int):
    """Remove a predicate from the session state."""
    if 0 <= index < len(st.session_state.predicates):
        st.session_state.predicates.pop(index)


def render_predicate_builder():
    """Render the predicate builder UI."""
    st.subheader("🔍 Filters (Predicates)")
    
    # Get available columns and operators
    available_columns = SearchService.get_available_predicate_columns()
    available_operators = SearchService.get_available_operators()
    
    # Display existing predicates
    if st.session_state.predicates:
        st.write("**Active Filters:**")
        for i, pred in enumerate(st.session_state.predicates):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text(f"{pred['key']} {pred['operator']} {pred['value']}")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    remove_predicate(i)
                    st.rerun()
    
    # Add new predicate
    st.write("**Add New Filter:**")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        selected_column = st.selectbox(
            "Column",
            options=available_columns,
            key="new_pred_column"
        )
    
    with col2:
        # Filter operators based on column type
        column_type = SearchService.get_predicate_type(selected_column)
        if column_type == 'date':
            default_operators = ["=", ">", "<", ">=", "<="]
        else:
            default_operators = available_operators
        
        selected_operator = st.selectbox(
            "Operator",
            options=default_operators,
            key="new_pred_operator"
        )
    
    with col3:
        # Render appropriate input based on column type
        column_type = SearchService.get_predicate_type(selected_column)
        
        if column_type == 'date':
            value = st.date_input(
                "Value",
                value=datetime.now(),
                key="new_pred_value"
            )
            value = value.strftime("%Y-%m-%d")
        elif column_type == 'int':
            value = st.number_input(
                "Value",
                value=0,
                step=1,
                key="new_pred_value"
            )
        else:
            value = st.text_input(
                "Value",
                key="new_pred_value"
            )
    
    with col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("➕ Add"):
            add_predicate(selected_column, selected_operator, value)
            st.rerun()
    
    # Clear all predicates button
    if st.session_state.predicates:
        if st.button("🗑️ Clear All Filters"):
            st.session_state.predicates = []
            st.rerun()


def render_search_form():
    """Render the main search form."""
    st.subheader("🔎 Search Query")
    
    # Query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., machine learning algorithms",
        key="search_query"
    )
    
    # Number of results
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input(
            "Number of results (k):",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            key="k_value"
        )
    
    with col2:
        method = st.selectbox(
            "Search method:",
            options=["post_search", "pre_search"],
            index=0,
            key="search_method",
            help="post_search: Filter after vector search\npre_search: Filter before vector search"
        )
    
    return query, k, method


def render_results(df: pd.DataFrame):
    """Render search results."""
    st.subheader("📊 Results")
    
    if df.empty:
        st.warning("No results found. Try adjusting your query or filters.")
        return
    
    st.write(f"Found **{len(df)}** results")
    
    # Display results with custom formatting
    for idx, row in df.iterrows():
        with st.expander(f"**{row.get('title', 'N/A')}** (Similarity: {row['similarity']:.4f})"):
            cols = st.columns(2)
            col_idx = 0
            for col_name, value in row.items():
                if col_name != 'title' and col_name != 'similarity':
                    with cols[col_idx % 2]:
                        st.write(f"**{col_name}:** {value}")
                    col_idx += 1
            
            # Show similarity prominently
            st.metric("Similarity Score", f"{row['similarity']:.4f}")
    
    # Also show as a table
    st.write("**Table View:**")
    st.dataframe(df, use_container_width=True)


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Hybrid Search",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🔍 Hybrid Search Interface")
    st.markdown("---")
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Search form
        query, k, method = render_search_form()
        
        # Predicate builder
        render_predicate_builder()
    
    with col2:
        # Search button
        st.subheader("🚀 Execute Search")
        search_button = st.button("Search", type="primary", use_container_width=True)
        
        if search_button:
            if not query:
                st.error("Please enter a search query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        # Create a new SearchService for each search to avoid threading issues
                        # SQLite connections are not thread-safe across different threads
                        with SearchService() as search_service:
                            # Perform search
                            results_df = search_service.perform_search(
                                query=query,
                                k=k,
                                predicates=st.session_state.predicates,
                                method=method
                            )
                        
                        # Store results in session state
                        st.session_state.last_results = results_df
                        st.success(f"Search completed! Found {len(results_df)} results.")
                        
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
    
    # Display results if available
    st.markdown("---")
    if 'last_results' in st.session_state and not st.session_state.last_results.empty:
        render_results(st.session_state.last_results)
    
    # Footer
    st.markdown("---")
    st.caption("Hybrid Search Interface - Combining vector similarity with metadata filtering")


if __name__ == "__main__":
    main()
