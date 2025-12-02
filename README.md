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

## Implementation Details

### Features

**Search Strategies:**
- **BASE_PRE_SEARCH**: Baseline pre-search filtering (filters DB first, then searches vectors with fixed nprobe)
- **BASE_POS_SEARCH**: Baseline post-search filtering (searches vectors first, then filters with fixed search_k)
- **ADAP_PRE_SEARCH**: Adaptive pre-search with dynamic nprobe calculation
- **ADAP_POS_SEARCH**: Adaptive post-search with survivor-based search_k optimization
- **LR_BASED_ADAP_SEARCH**: **Machine Learning based strategy selection** (automatically chooses between PRE/POS)

**ML Based Strategy Selection:**
- **Logistic Regression Model**: Trained on real query performance data
- **Hardcoded Parameters**: Zero pkl loading overhead for maximum inference speed
- **Features**: k, num_survivors, selectivity
- **Performance**: ~96% F1 score, <0.5ms inference time
- **Auto-selection**: Predicts whether PRE or POS will be faster and executes accordingly

**Smart Parameter Optimization:**
- `estimate_survivors()`: Uses 2D histograms to estimate predicate selectivity
- Adaptive `nprobe` calculation based on survivor count (PRE search)
- Adaptive `search_k` calculation based on selectivity (POS search)
- Exponential search expansion for iterative refinement
- Dynamic parameter tuning until k results are found

### Model Training & Evaluation

**Model Training Pipeline (`model_evaluation.py`):**
1. Loads timing data from `timed_results.csv`
2. Trains 4 models:
   - Simple Rule-Based (threshold on num_survivors)
   - Advanced Rule-Based (k-dependent thresholds)
   - **Logistic Regression**
   - Decision Tree
3. Evaluates each model on:
   - Accuracy, Precision, Recall, F1 Score
   - **Inference time** (measured with `time.perf_counter()`)
4. Selects best model using composite score (70% F1 + 20% Accuracy + 10% Speed)
5. **Automatically injects** model parameters into `search.py` as hardcoded values

**Model Performance:**
| Model | Accuracy | F1 Score | Inference Time | Composite Score |
|-------|----------|----------|----------------|-----------------|
| **Logistic Regression** | **~0.95** | **~0.96** | **~0.5 ms** | **~0.95**  |
| Decision Tree | ~0.94 | ~0.95 | ~0.8 ms | ~0.94 |
| Advanced Rule-Based | ~0.90 | ~0.91 | ~2.0 ms | ~0.88 |
| Simple Rule-Based | ~0.85 | ~0.86 | ~0.1 ms | ~0.83 |

## 💻 Usage Examples

### ML-Based Adaptive Search

The **fastest and smartest** way to search - automatically selects the optimal strategy:

```python
from search import Search
from shared_dataclasses import Predicate

# Initialize search engine
search = Search()

# Prepare query
query = "machine learning algorithms"
query_embedding = search.embedder.encode_query(query)

# Define predicates
predicates = [
    Predicate(key="token_count", value=400, operator=">="),
]

# LR-based adaptive search - automatically chooses PRE or POS!
results = search.lr_based_adap_search(query_embedding, predicates, k=10)

print(f"Found {len(results.results)} results")
print(results.to_df(show_cols=['item_id', 'title']))

search.close()
```

### Manual Strategy Selection

For explicit control over search strategy:

```python
from search import Search
from shared_dataclasses import Predicate
from timer import Timer, SearchMethod

# Initialize with timer for performance tracking
timer = Timer()
search = Search(timer=timer)

# Prepare query
query = "deep learning"
query_embedding = search.embedder.encode_query(query)
predicates = [
    Predicate(key="token_count", value=500, operator="<"),
    Predicate(key="revdate", value="2025-01-01", operator=">=")
]

# Method 1: Use search() with method parameter
results = search.search(query, predicates, k=5, method=SearchMethod.ADAP_PRE_SEARCH)

# Method 2: Call strategy functions directly
est_survivors = search.histo.estimate_survivors(predicates)

# Adaptive PRE-search
pre_results = search.adap_pre_search(query_embedding, predicates, k=5)

# Adaptive POS-search
pos_results = search.adap_pos_search(query_embedding, predicates, k=5, est_survivors)

# Get timing stats
timing_stats = timer.get_stats()
print(f"Total time: {timing_stats['total']:.2f} ms")

search.close()
```

### Training the ML Model

To retrain the model with new performance data:

```bash
# 1. Collect timing data (run your benchmark to generate timed_results.csv)
# 2. Train models and inject parameters into search.py
python model_evaluation.py
```

**This will:**
- Train 4 different models (Rule-Based, LR, Decision Tree)
- Measure inference time for each model
- Select the best model (usually Logistic Regression)
- **Automatically update search.py** with hardcoded parameters
- Generate visualizations (model_comparison.png, confusion_matrices.png, decision_tree.png)

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
- **ML Model Inference**: <0.5ms prediction time 
- **Strategy Selection Accuracy**: ~96% F1 score in choosing optimal strategy
- **Memory Usage**: Streaming processing for large datasets
- **Storage**: Compressed FAISS index + lightweight SQLite metadata
- **Adaptive Tuning**: Dynamic parameter optimization based on query selectivity and ML predictions

## 📁 File Structure

```
hybrid_search/
├── datafetch.py              # Data collection script
├── add_item_id_to_csv.py     # ID assignment utility
├── dbManagement.py           # SQLite database creation
├── create_faiss_from_csv.py  # FAISS index builder
├── search.py                 # Core search engine (with LR model)
├── vector_index.py           # Vector operations
├── histo2d.py                # 2D histogram for selectivity estimation
├── timer.py                  # Performance timing utilities
├── shared_dataclasses.py     # Common data structures
├── settings.py               # Configuration
│
├── model_evaluation.py       # ML model training & evaluation
├── train_and_save_model.py.  # Alternative: saves model as pkl (optional)
├── example_lr_search.py      # Usage examples for LR-based search
│
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment
├── install.sh                # Automated installation script
├── setup_workflow.sh         # Automated workflow setup script
│
├── test_hybrid_strategies.py # Strategy comparison tests
├── test_vector_index.py     # Vector index tests
├── test_db_search.py        # Database search tests
│
├── timed_results.csv         # Performance data for model training
├── meta_wiki.db              # Metadata database
├── wikipedia_sample_150k_with_ids.csv  # Processed data
│
├── model_comparison.png      # Generated: Model performance comparison
├── confusion_matrices.png    # Generated: Model confusion matrices
├── decision_tree.png         # Generated: Decision tree visualization
│
├── README.md                 # This file
│
└── index.faiss.new/          # FAISS index files
    ├── index_meta.json
    └── vectors.index
```

## 🔬 Advanced Features

- **ML-Based Strategy Selection**: Logistic Regression model predicts optimal search strategy
- **Zero-Overhead Inference**: Hardcoded model parameters eliminate pkl file I/O
- **Comprehensive Model Evaluation**: Compares 4 models on accuracy and inference speed
- **Automatic Parameter Injection**: Updates search.py automatically after training
- **Reservoir Sampling**: Memory-efficient sampling for index training
- **Chunked Processing**: Handles datasets larger than available RAM
- **Index Persistence**: Save/load trained indexes for reuse
- **Metadata Normalization**: Handles complex data types and missing values
- **Adaptive Query Optimization**: Combines vector and metadata queries efficiently
- **2D Histogram Estimation**: Sophisticated selectivity estimation for query planning
- **Iterative Search Refinement**: Dynamic parameter tuning during search execution
- **Performance Timing**: Built-in timer for detailed performance analysis

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

This project implements ML-optimized adaptive hybrid search strategies. 

1. All baseline and adaptive search strategies (BASE_PRE, BASE_POS, ADAP_PRE, ADAP_POS)
2. ML-based automatic strategy selection (Logistic Regression)
3. Comprehensive model evaluation framework
4. Zero-overhead inference with hardcoded parameters
5. Performance benchmarking and timing utilities
6. 2D histogram-based selectivity estimation

