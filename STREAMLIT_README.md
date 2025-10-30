# Hybrid Search Streamlit Interface

A minimalist front-end interface for the hybrid search system, built with Streamlit.

## Features

- **Text Query Input**: Enter natural language search queries
- **Configurable Results (k)**: Specify the number of results to return (1-1000)
- **Dynamic Predicate Filters**: Add multiple filters with:
  - Column selection (item_id, revdate, token_count)
  - Operator selection (=, >, <, >=, <=, IN)
  - Type-appropriate value inputs (date picker for dates, number input for integers)
- **Search Method Selection**: Choose between pre-search and post-search strategies
- **Results Display**: 
  - Expandable cards showing similarity scores
  - Table view for easy scanning
  - Highlighted similarity metrics

## Architecture

The application maintains strict separation between frontend and backend:

- **`app.py`**: Streamlit frontend - handles UI, user interactions, and display
- **`backend.py`**: Backend service layer - wraps search functionality and provides clean API
- **`search.py`**: Core search logic - vector similarity and hybrid search strategies

## Running the Application

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Basic Search

1. Enter a search query (e.g., "machine learning algorithms")
2. Set the number of results (k)
3. Click "Search"

### Adding Filters

1. Select a column from the dropdown (item_id, revdate, or token_count)
2. Choose an operator (=, >, <, >=, <=)
3. Enter the value:
   - For **revdate**: Use the date picker
   - For **item_id** and **token_count**: Enter a number
4. Click "➕ Add" to add the filter
5. Repeat to add multiple filters (they are combined with AND logic)

### Removing Filters

- Click the "❌" button next to a specific filter to remove it
- Click "🗑️ Clear All Filters" to remove all filters at once

### Search Methods

- **post_search** (recommended): Performs vector search first, then applies filters
- **pre_search**: Applies filters first, then performs vector search on filtered results

## Example Queries

1. **General Search**:
   - Query: "artificial intelligence"
   - k: 10
   - No filters

2. **Date-Filtered Search**:
   - Query: "climate change"
   - k: 5
   - Filter: revdate >= 2025-01-01

3. **Multi-Filter Search**:
   - Query: "quantum computing"
   - k: 10
   - Filter 1: token_count > 100
   - Filter 2: token_count < 500
   - Filter 3: revdate >= 2025-01-15

## Technical Details

### Predicate Columns

- **item_id**: Integer - Unique identifier for each article
- **revdate**: Date - Revision date of the article
- **token_count**: Integer - Number of tokens in the article

### Operators

- **=**: Equal to
- **>**: Greater than
- **<**: Less than
- **>=**: Greater than or equal to
- **<=**: Less than or equal to
- **IN**: In list (for multiple values)

### Search Results

Results include:
- **Title**: Article title
- **Similarity Score**: Vector similarity (0-1, higher is better)
- **Metadata**: item_id, revdate, token_count, and other available fields

## Troubleshooting

### Application won't start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that the FAISS index exists at the path specified in `settings.py`
- Verify the database file exists at the path specified in `settings.py`

### No results found

- Try simplifying your query
- Remove or relax filters
- Check that filters use valid values (e.g., dates in YYYY-MM-DD format)

### Slow searches

- Reduce the value of k
- Use pre_search method if filters are highly selective
- Use post_search method if filters are less selective

## Development

To modify the interface:

1. **Frontend changes**: Edit `app.py`
2. **Backend logic**: Edit `backend.py`
3. **Core search**: Edit `search.py`

The modular architecture makes it easy to swap out components or add new features.
