# Hybrid Search System

A complete hybrid search system that combines semantic vector search with metadata filtering. Built with FAISS for efficient vector similarity search and SQLite for metadata storage.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Wikipedia     │ => │   CSV with IDs   │ => │   SQLite DB     │
│   Dataset       │    │   & Metadata     │    │   (Metadata)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   FAISS Index    │ => │   Hybrid        │
                       │   (Vectors)      │    │   Search        │
                       └──────────────────┘    └─────────────────┘
```

## Features

- **Semantic Search**: Uses sentence transformers to encode text into vectors
- **Vector Indexing**: FAISS IVF-PQ index for fast similarity search
- **Metadata Filtering**: SQLite database for structured metadata queries
- **Hybrid Queries**: Combine semantic similarity with metadata filters
- **Scalable**: Processes large datasets with streaming and chunked operations

## Pipeline Components

### 1. Data Collection (`datafetch.py`)

Fetches Wikipedia articles from HuggingFace datasets and creates the initial CSV.

```bash
# Fetch 150k articles (default)
python datafetch.py

# Fetch custom number of articles
python datafetch.py 50000
```

**Settings:**
- `REPO_ID`: HuggingFace dataset identifier
- `N`: Target number of articles (default: 150,000)
- `CHUNK_FETCH`: Articles per chunk for shuffling
- `OUT_CSV`: Output CSV filename

**Output:** `wikipedia_sample_150k_with_ids.csv` with columns:
- `item_id`: Sequential ID (added later)
- `id`: Original Wikipedia article ID
- `title`: Article title
- `text`: Article content
- `categories`: Article categories (semicolon-separated)
- `url`: Wikipedia URL
- `revdate`: Revision date
- `token_count`: Number of tokens
- `entity`: Named entities

### 2. ID Assignment (`add_item_id_to_csv.py`)

Adds sequential item IDs to the CSV for FAISS index mapping.

```bash
python add_item_id_to_csv.py wikipedia_sample_150k.csv [output_file]
```

**Process:**
1. Reads the CSV file
2. Inserts `item_id` column with sequential integers (0, 1, 2, ...)
3. Creates new CSV with `_with_ids` suffix

**Output:** `wikipedia_sample_150k_with_ids.csv`

### 3. Metadata Database (`dbManagement.py`)

Creates SQLite database with normalized metadata for efficient filtering.

```bash
python dbManagement.py
```

**Database Schema:**
- **`items` table**: Core article metadata
  - `item_id`, `ext_id`, `title`, `url`, `revdate`, `token_count`, `entity`
- **`item_categories` table**: Normalized categories (many-to-many)
  - `item_id`, `category`

**Process:**
1. Creates fresh database with proper schema
2. Loads CSV data in chunks for memory efficiency
3. Processes categories into separate table
4. Creates indexes for query performance

**Settings:**
- `USECOLS`: Columns to load from CSV
- `DBCOLS`: Column mapping for database
- `CHUNK_SIZE`: Batch size for processing

### 4. Vector Index (`create_faiss_from_csv.py`)

Creates FAISS IVF-PQ index for fast vector similarity search.

```bash
python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv [train_samples] [nlist]
```

**Process:**
1. **Training Phase**: Samples texts for index training
2. **Embedding**: Encodes texts using sentence transformers
3. **Normalization**: L2 normalization for cosine similarity
4. **Index Training**: Creates IVF-PQ index with training vectors
5. **Vector Addition**: Adds all vectors to index in chunks

**Parameters:**
- `train_samples`: Number of texts for training (default: 50,000)
- `nlist`: Number of IVF clusters (default: 4,096)
- `pq_m`: Product quantization sub-vectors (default: 32)
- `pq_bits`: Bits per sub-vector (default: 8)

**Output:**
- FAISS index files (`.index` and `.json` metadata)
- Index statistics and performance metrics

## Usage Examples

### Basic Setup

```bash
# 1. Fetch data
python datafetch.py 150000

# 2. Add IDs
python add_item_id_to_csv.py wikipedia_sample_150k.csv

# 3. Create metadata database
python dbManagement.py

# 4. Create vector index
python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv 50000 4096
```

### Search System (`main.py`)

```python
from hybrid_search import HybridSearch

# Initialize search system
search = HybridSearch(
    db_path="meta_wiki.db",
    index_path="faiss_index_from_csv_nlist4096_pq32x8"
)

# Semantic search
results = search.search(
    query="machine learning algorithms",
    k=10
)

# Hybrid search with metadata filters
results = search.search(
    query="artificial intelligence",
    k=10,
    filters={
        "min_token_count": 100,
        "categories": ["Technology", "Computer Science"]
    }
)

# Get article by ID
article = search.get_article(12345)
```

## Configuration (`settings.py`)

```python
# Data settings
N = 150_000                    # Target dataset size
REPO_ID = "DragonLLM/Clean-Wikipedia-English-Articles"
OUT_CSV = "wikipedia_sample_150k_with_ids.csv"

# Database settings
DB_PATH = Path("meta_wiki.db")
CHUNK_SIZE = 5_000            # Processing batch size

# Vector index settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLIST = 4096                  # IVF clusters
PQ_M = 32                     # PQ sub-vectors
PQ_BITS = 8                   # Bits per sub-vector
TRAIN_MAX = 50_000           # Training samples
```

## Performance Characteristics

- **Vector Search**: Sub-second similarity queries on millions of vectors
- **Metadata Filtering**: Fast SQL queries with proper indexing
- **Memory Usage**: Streaming processing for large datasets
- **Storage**: Compressed FAISS index + lightweight SQLite metadata

## File Structure

```
hybrid_search/
├── datafetch.py              # Data collection script
├── add_item_id_to_csv.py     # ID assignment utility
├── dbManagement.py           # SQLite database creation
├── create_faiss_from_csv.py  # FAISS index builder
├── main.py                   # Main search interface
├── vector_index.py           # Vector operations
├── settings.py               # Configuration
├── meta_wiki.db              # Metadata database
├── wikipedia_sample_150k_with_ids.csv  # Processed data
└── index.faiss/              # FAISS index files
    ├── index_meta.json
    └── vectors.index
```

## Requirements

- Python 3.8+
- FAISS
- Sentence Transformers
- Pandas
- SQLite3
- HuggingFace Datasets

## Advanced Features

- **Reservoir Sampling**: Memory-efficient sampling for index training
- **Chunked Processing**: Handles datasets larger than available RAM
- **Index Persistence**: Save/load trained indexes for reuse
- **Metadata Normalization**: Handles complex data types and missing values
- **Query Optimization**: Combines vector and metadata queries efficiently
