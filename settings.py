# project settings
from pathlib import Path

SEED = 6400
N = 150_000


# for data loading
REPO_ID = "DragonLLM/Clean-Wikipedia-English-Articles"
CHUNK_FETCH = 50_000     # fetch this many, shuffle locally, then write
LOG_EVERY = 2_000
OUT_CSV = "wikipedia_sample_150k.csv"
COLUMNS = ["id","title","text","categories","url","revdate","token_count","entity"]

# for sqlite database creation
DB_PATH = Path("meta_wiki.db")
USECOLS = ["id", "title", "categories", "url", "revdate", "token_count", "entity"]
CHUNK_SIZE = 20_000

# IVF settings
NLIST        = 4096                 # #clusters ~ sqrt(N)
PQ_M         = 32                   # #subquantizers
PQ_BITS      = 8                    # bits per subquantizer
TRAIN_MAX    = 50_000               # max vectors to train on (sampled)