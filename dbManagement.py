import sqlite3
import numpy as np
from settings import DB_PATH, USECOLS, ITEM_COLS_DB, ITEM_COLS_CSV, CATEGORY_COLS_DB, CATEGORY_COLS_CSV
import pandas as pd
from typing import Iterator, List

class DbManagement:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cur = self.conn.cursor()

    def create_db(self):
        # if a DB exists, close current connection and remove the file
        try:
            self.conn.close()
        except Exception:
            pass
        if DB_PATH.exists():
            DB_PATH.unlink()
        # because we define create here, we always open a fresh connection + cursor
        self.conn = sqlite3.connect(DB_PATH)
        self.cur = self.conn.cursor()
        self.cur.execute("PRAGMA journal_mode = WAL;")
        self.cur.execute("""
        CREATE TABLE items (
            item_id INTEGER PRIMARY KEY,
            ext_id INTEGER,
            title TEXT,
            text TEXT,
            url TEXT,
            revdate TEXT,
            token_count INTEGER,
            entity TEXT
        );
        """)
        self.cur.execute("PRAGMA foreign_keys = ON;")
        ### to do: maybe we should drop categories, the values do not make sense. consider synthetic data if we want to categorical predicates.
        self.cur.execute("""
        CREATE TABLE item_categories (
            item_id INTEGER,
            category TEXT,
            PRIMARY KEY (item_id, category),
            FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
        );
        """)
        self.conn.commit()

    def load_df_to_db(self, df: pd.DataFrame):
        # item_id column already exists in the CSV, no need to insert
        df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).astype("int64")

        # Process dates more efficiently
        rev = pd.to_datetime(df["revdate"], errors="coerce")
        df["revdate"] = rev.dt.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare data for database insertion
        items_df = df[ITEM_COLS_CSV].copy()
        items_df.columns = ITEM_COLS_DB
        items_df["ext_id"] = items_df["ext_id"].astype("int64", errors="ignore")

        # Insert items into database
        items_df.to_sql("items", self.conn, if_exists="append", index=False)

        # Process and insert categories (need to load categories separately)
        categories_data = []
        for row in df[CATEGORY_COLS_CSV].itertuples(index=False):
            item_id, categories = row
            for cat in self._split_cats(categories):
                categories_data.append((int(item_id), cat))
        if categories_data:
            self.cur.executemany(
                f"INSERT OR IGNORE INTO item_categories ({CATEGORY_COLS_DB[0]}, {CATEGORY_COLS_DB[1]}) VALUES (?, ?);",
                categories_data,
            )
        self.conn.commit()

    def load_db(
        self,
        reader: Iterator[pd.DataFrame],
    ) -> None:
        for df in reader:
            df = df.copy()
            self.load_df_to_db(df)
        self._create_indexes()
        self.cur.execute("SELECT COUNT(*) FROM items;")
        print(f"Loaded {self.cur.fetchone()[0]} items into the database")

    def stream_items_for_index(self, text_col: str, batch_size: int = 20000):
        last_id = -1
        while True:
            self.cur.execute(
                f"SELECT item_id, {text_col} FROM items WHERE item_id > ? ORDER BY item_id LIMIT ?;",
                (last_id, batch_size),
            )
            rows = self.cur.fetchall()
            if not rows:
                break
            ids = np.asarray([r[0] for r in rows], dtype="int64")
            texts = [(r[1] if r[1] is not None else "") for r in rows]
            yield ids, texts
            last_id = int(ids[-1])

    def _create_indexes(self):
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_items_revdate     ON items(revdate);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_items_tokencount  ON items(token_count);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_items_entity      ON items(entity);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_cats_category     ON item_categories(category);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_cats_item         ON item_categories(item_id);")
        self.conn.commit()
        self.cur.execute("PRAGMA synchronous = NORMAL;")

    def _split_cats(self, s: str) -> List[str]:
        if pd.isna(s) or not isinstance(s, str):
            return []
        return [c.strip() for c in s.split(";") if c.strip()]


if __name__ == "__main__":
    # for initializing the database
    db_management = DbManagement()
    db_management.create_db()
    from settings import OUT_CSV, CHUNK_SIZE
    reader = pd.read_csv(OUT_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    db_management.load_db(reader)