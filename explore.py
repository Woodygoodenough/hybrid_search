# %% imports
import pandas as pd
import sqlite3
from pathlib import Path
from itertools import islice

# --- settings ---
CSV_PATH = "wikipedia_sample_150k.csv"
CHUNK_SIZE = 20_000   # tune for your RAM / SSD
DB_PATH = Path("meta_wiki.db")
USECOLS = ["id", "title", "categories", "url", "revdate", "token_count", "entity"]

# --- dev settings ---
N_chunks = 1


# %% 1) load slice

# Normalize categories into list[str]
def split_cats(s):
    if pd.isna(s) or not isinstance(s, str):
        return []
    return [c.strip() for c in s.split(";") if c.strip()]


# %% 2) create SQLite (fresh)
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Pragmas for fast bulk load (reset later if you want durability over speed)
cur.execute("PRAGMA journal_mode = WAL;")
cur.execute("PRAGMA synchronous = OFF;")
cur.execute("PRAGMA temp_store = MEMORY;")
cur.execute("PRAGMA cache_size = -200000;")  # ~200MB cache

# items table: metadata only, no 'text'
cur.execute("""
CREATE TABLE items (
    item_id      INTEGER PRIMARY KEY,   -- stable 0..N-1
    ext_id       INTEGER,               -- original CSV 'id'
    title        TEXT,
    url          TEXT,
    revdate      TEXT,                  -- stored as ISO string for simplicity
    token_count  INTEGER,
    entity       TEXT
);
""")
# Many-to-many table for categories
cur.execute("PRAGMA foreign_keys = ON;")
cur.execute("""
CREATE TABLE item_categories (
    item_id   INTEGER,
    category  TEXT,
    PRIMARY KEY (item_id, category),
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);
""")
conn.commit()
next_item_id = 0

reader = pd.read_csv(
    CSV_PATH,
    usecols=USECOLS,
    chunksize=CHUNK_SIZE,
    dtype={"id": "Int64", "token_count": "Int64", "title": "string", "url": "string", "entity": "string"},
    # parse revdate as string then to datetime per-chunk; avoids mixed formats surprises
)

for df in islice(reader, N_chunks):
    # minimal clean-up per chunk
    df = df.copy()
    n = len(df)
    df.insert(0, "item_id", range(next_item_id, next_item_id + n))
    next_item_id += n

    df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).astype("int64")
    # parse -> iso string
    rev = pd.to_datetime(df["revdate"], errors="coerce")
    df["revdate_iso"] = rev.dt.strftime("%Y-%m-%d %H:%M:%S")

    # write items (metadata only)
    items_df = pd.DataFrame({
        "item_id": df["item_id"],
        "ext_id": df["id"].astype("int64", errors="ignore"),
        "title": df["title"],
        "url": df["url"],
        "revdate": df["revdate_iso"],
        "token_count": df["token_count"],
        "entity": df["entity"],
    })

    # Use to_sql with method='multi' to batch rows
    items_df.to_sql("items", conn, if_exists="append", index=False, method="multi", chunksize=10_000)

    # stream categories without creating a huge intermediate
    # build a generator of tuples (item_id, category)
    cat_pairs = (
        (int(row.item_id), cat)
        for row in df[["item_id", "categories"]].itertuples(index=False)
        for cat in split_cats(row.categories)
    )

    # executemany in manageable batches
    batch = []
    B = 50_000
    for t in cat_pairs:
        batch.append(t)
        if len(batch) >= B:
            cur.executemany(
                "INSERT OR IGNORE INTO item_categories (item_id, category) VALUES (?, ?);",
                batch
            )
            batch.clear()
    if batch:
        cur.executemany(
            "INSERT OR IGNORE INTO item_categories (item_id, category) VALUES (?, ?);",
            batch
        )

    conn.commit()  # commit per chunk to keep WAL file bounded
# 3) Create indexes AFTER the load (much faster)
cur.execute("CREATE INDEX IF NOT EXISTS idx_items_revdate     ON items(revdate);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_items_tokencount  ON items(token_count);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_items_entity      ON items(entity);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_cats_category     ON item_categories(category);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_cats_item         ON item_categories(item_id);")
conn.commit()

# Optional: tighten durability after bulk load
cur.execute("PRAGMA synchronous = NORMAL;")

# quick counts
for tbl in ("items", "item_categories"):
    print(tbl, cur.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0])

conn.execute("PRAGMA wal_checkpoint(FULL);")
conn.close()
print(f"Done. DB at {DB_PATH.resolve()}")