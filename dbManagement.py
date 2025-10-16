import sqlite3
from settings import DB_PATH
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

    def load_df_to_db(self, df: pd.DataFrame, start_item_id: int, end_item_id: int):
        df = df.copy()
        df.insert(0, "item_id", range(start_item_id, end_item_id))
        df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).astype("int64")
        # parse -> iso string
        rev = pd.to_datetime(df["revdate"], errors="coerce")
        df["revdate_iso"] = rev.dt.strftime("%Y-%m-%d %H:%M:%S")
        items_df = pd.DataFrame({
            "item_id": df["item_id"],
            "ext_id": df["id"].astype("int64", errors="ignore"),
            "title": df["title"],
            "url": df["url"],
            "revdate": df["revdate_iso"],
            "token_count": df["token_count"],
            "entity": df["entity"],
        })
        items_df.to_sql("items", self.conn, if_exists="append", index=False, method="multi", chunksize=10_000)
        # stream categories without creating a huge intermediate
        # build a generator of tuples (item_id, category)
        cat_pairs = (
            (int(row.item_id), cat)
            for row in df[["item_id", "categories"]].itertuples(index=False)
            for cat in self._split_cats(row.categories)
        )
        batch = []
        B = 50_000
        for t in cat_pairs:
            batch.append(t)
            if len(batch) >= B:
                self.cur.executemany(
                    "INSERT OR IGNORE INTO item_categories (item_id, category) VALUES (?, ?);",
                    batch
                )
                batch.clear()
        if batch:
            self.cur.executemany(
                "INSERT OR IGNORE INTO item_categories (item_id, category) VALUES (?, ?);",
                batch
            )
        self.conn.commit()


    def load_db(self, reader: Iterator[pd.DataFrame]):
        next_item_id = 0
        for df in reader:
            # minimal clean-up per chunk
            df = df.copy()
            end_item_id = next_item_id + len(df)
            self.load_df_to_db(df, next_item_id, end_item_id)
            next_item_id = end_item_id
            print(f"loaded {next_item_id} items into database")
        self._create_indexes()
        # sanity check: count items
        self.cur.execute("SELECT COUNT(*) FROM items;")
        count = self.cur.fetchone()[0]
        if count != next_item_id:
            raise ValueError(f"Expected {next_item_id} items, got {count}")

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