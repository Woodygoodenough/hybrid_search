# Hybrid Search System

A complete hybrid search system that combines semantic vector search with metadata filtering. Built with FAISS for efficient vector similarity search and SQLite for metadata storage.

## 🚀 Quick Start

### Installation

#### Option 1: Automated Installation
```bash
# Ignore this, we can do this in the final distribution
./install.sh
```

#### Option 2: Using Conda
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate hybrid_search
```

#### Option 3: Using Pip
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Option 4: Manual Installation
```bash
# Core dependencies
pip install faiss-cpu sentence-transformers numpy pandas datasets

```

### Complete Workflow Setup

#### Option 1: Automated Setup
```bash
# Run the complete workflow automatically (not sure if it works. Recommend using the manual setup)
./setup_workflow.sh
```

#### Option 2: Manual Setup
```bash
# 1. Fetch Wikipedia data (150k articles)
python datafetch.py

# 2. Add sequential IDs to CSV
python add_item_id_to_csv.py wikipedia_sample_150k.csv

# 3. Create metadata database
python dbManagement.py

# 4. Build FAISS vector index
python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv

```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Wikipedia     │ => │   CSV with IDs   │ => │   SQLite DB     │
│   Dataset       │    │   & Metadata     │    │   (Metadata)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   FAISS Index    │ => │   Hybrid        │
                       │   (Vectors)      │    │   Search        │
                       └──────────────────┘    └─────────────────┘
```

## 📋 Pipeline Components

### 1. Data Collection (`datafetch.py`)

Fetches Wikipedia articles from HuggingFace datasets using streaming and chunked processing.

```bash
# Fetch 150k articles (default)
python datafetch.py

# Fetch custom number of articles
python datafetch.py 50000
```

**Features:**
- Streaming data processing for memory efficiency
- Local shuffling for better data distribution
- Progress tracking and ETA estimation
- Configurable chunk sizes

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

### 3. Metadata Database (`dbManagement.py`)

Creates SQLite database with normalized metadata for efficient filtering.

```bash
python dbManagement.py
```

**Database Schema:**
- **`items` table**: Core article metadata
  - `item_id`, `ext_id`, `title`, `url`, `revdate`, `token_count`, `entity`

**Features:**
- Chunked processing for large datasets
- Automatic date normalization
- Indexed columns for fast queries
- WAL mode for better concurrency

### 4. Vector Index (`create_faiss_from_csv.py`)

Creates FAISS IVF-PQ index for fast vector similarity search.

```bash
python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv [train_samples] [nlist]
```

**Process:**
1. **Training Phase**: Samples texts for index training using reservoir sampling
2. **Embedding**: Encodes texts using sentence transformers
3. **Normalization**: L2 normalization for cosine similarity
4. **Index Training**: Creates IVF-PQ index with training vectors
5. **Vector Addition**: Adds all vectors to index in chunks

**Parameters:**
- `train_samples`: Number of texts for training (default: 60,000)
- `nlist`: Number of IVF clusters (default: 512)
- `pq_m`: Product quantization sub-vectors (default: 32)
- `pq_bits`: Bits per sub-vector (default: 8)

## 🔍 Current Implementation Status

### ✅ Implemented Features

**Adaptive Search Strategies:**
- **Pre-search filtering**: Filters database first, then searches vectors
- **Post-search filtering**: Searches vectors first, then filters results
- **Adaptive nprobe**: Dynamically adjusts search parameters based on expected survivors
- **Histogram-based estimation**: Uses 2D histograms to estimate predicate selectivity

**Smart Parameter Optimization:**
- `estimate_survivors()`: Estimates how many records match given predicates
- Adaptive `nprobe` calculation based on selectivity
- Exponential search expansion for post-search strategy
- Iterative refinement until k results are found

### 🚧 Current Stage: Adaptive Pre/Post Search

**What's Working:**
- Both pre-search and post-search strategies are implemented
- Adaptive parameter tuning based on `estimate_survivors`
- Proper threshold handling for different selectivity scenarios
- Comprehensive test suite validating both strategies

**What's Missing:**
- **Baseline approaches**: Coarse, non-adaptive versions of both strategies
- **Strategy merging**: Unified approach combining pre/post benefits
- **Performance benchmarking**: Comparison between adaptive and baseline methods

### 🎯 Next Steps

1. **Implement Baseline Approaches:**
   - Coarse pre-search: Fixed nprobe, no adaptive tuning
   - Coarse post-search: Fixed search parameters, no survivor estimation
   - These will serve as performance baselines

2. **Strategy Selection:**
   - Implement threshold-based strategy selection
   - Use `estimate_survivors` to choose optimal approach
   - Merge strategies for hybrid benefits

3. **Performance Optimization:**
   - Benchmark adaptive vs baseline approaches
   - Fine-tune threshold parameters
   - Optimize histogram resolution

## 💻 Usage Examples

### Interactive Search (`main.py`)
Not implemented yet


### Programmatic Search (`search.py`)

For integration into other applications:

```python
from search import Search
from shared_dataclasses import Predicate

# Initialize search engine
with Search() as search:
    # Simple search
    results = search.search("machine learning algorithms", [], k=5)
    
    # Search with metadata filters
    predicates = [
        Predicate(key="token_count", value=500, operator="<"),
        Predicate(key="revdate", value="2025-01-01", operator=">=")
    ]
    results = search.search("AI research", predicates, k=5)
    
    # Choose search strategy
    pre_results = search.search("deep learning", predicates, k=5, method="pre_search")
    post_results = search.search("deep learning", predicates, k=5, method="post_search")
```

### Testing (`test_hybrid_strategies.py`)

Run comprehensive tests for both search strategies:

```bash
python test_hybrid_strategies.py
```

Tests cover:
- Exact k matches and boundary conditions
- Edge cases (k=1, empty results, very large k)
- Single and multiple predicate combinations
- Different query terms
- Performance comparison between strategies

## ⚙️ Configuration (`settings.py`)

```python
# Data settings
N = 150_000                    # Target dataset size
REPO_ID = "DragonLLM/Clean-Wikipedia-English-Articles"
OUT_CSV = "wikipedia_sample_150k_with_ids.csv"

# Database settings
DB_PATH = Path("meta_wiki.db")
CHUNK_SIZE = 10_000           # Processing batch size

# Vector index settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLIST = 512                   # IVF clusters
PQ_M = 32                     # PQ sub-vectors
PQ_BITS = 8                   # Bits per sub-vector
TRAIN_MAX = 60_000           # Training samples
NPROBE = 16                   # Default nprobe
N_PER_CLUSTER = 30_000       # Estimated items per cluster
```

## 📊 Performance Characteristics

- **Vector Search**: Sub-second similarity queries on 150k vectors
- **Metadata Filtering**: Fast SQL queries with proper indexing
- **Memory Usage**: Streaming processing for large datasets
- **Storage**: Compressed FAISS index + lightweight SQLite metadata
- **Adaptive Tuning**: Dynamic parameter optimization based on query selectivity

## 📁 File Structure

```
hybrid_search/
├── datafetch.py              # Data collection script
├── add_item_id_to_csv.py     # ID assignment utility
├── dbManagement.py           # SQLite database creation
├── create_faiss_from_csv.py  # FAISS index builder
├── search.py                 # Core search engine
├── vector_index.py           # Vector operations
├── histo2d.py                # 2D histogram for selectivity estimation
├── shared_dataclasses.py     # Common data structures
├── settings.py               # Configuration
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment
├── install.sh                # Automated installation script
├── setup_workflow.sh         # Automated workflow setup script
├── test_hybrid_strategies.py # Strategy comparison tests
├── test_vector_index.py     # Vector index tests
├── test_db_search.py        # Database search tests
├── meta_wiki.db              # Metadata database
├── wikipedia_sample_150k_with_ids.csv  # Processed data
└── index.faiss.new/          # FAISS index files
    ├── index_meta.json
    └── vectors.index
```

## 🔬 Advanced Features

- **Reservoir Sampling**: Memory-efficient sampling for index training
- **Chunked Processing**: Handles datasets larger than available RAM
- **Index Persistence**: Save/load trained indexes for reuse
- **Metadata Normalization**: Handles complex data types and missing values
- **Adaptive Query Optimization**: Combines vector and metadata queries efficiently
- **2D Histogram Estimation**: Sophisticated selectivity estimation for query planning
- **Iterative Search Refinement**: Dynamic parameter tuning during search execution

## 🧪 Testing

The project includes comprehensive test suites:

```bash
# Test hybrid search strategies
python test_hybrid_strategies.py

# Test vector index functionality
python test_vector_index.py

# Test database search operations
python test_db_search.py
```

## 🤝 Contributing

This project implements adaptive hybrid search strategies with sophisticated parameter optimization. The current focus is on:

1. Implementing baseline (non-adaptive) approaches for comparison
2. Developing strategy selection mechanisms
3. Performance benchmarking and optimization
