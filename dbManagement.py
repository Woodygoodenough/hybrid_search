from dataclasses import dataclass
import sqlite3
import numpy as np
from settings import DB_PATH, USECOLS, ITEM_COLS_DB, ITEM_COLS_CSV, PRIMARY_KEY, TABLE_NAME, DATE_COLUMNS
import pandas as pd
from typing import Iterator, List, Tuple

@dataclass
class ArticleRecord:
    item_id: int
    ext_id: int
    title: str
    url: str
    revdate: str
    token_count: int
    entity: str

    @classmethod
    def from_row(cls, row: Tuple[int, int, str, str, str, int, str]) -> "ArticleRecord":
        return cls(item_id=row[0], ext_id=row[1], title=row[2], url=row[3], revdate=row[4], token_count=row[5], entity=row[6])

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
        sql_prep=f"CREATE TABLE {TABLE_NAME} ("
        for col in ITEM_COLS_DB:
            sql_prep += f"{col} TEXT, "
        sql_prep += f"PRIMARY KEY ({PRIMARY_KEY}))"
        self.cur.execute(sql_prep)
        self.conn.commit()

    def load_df_to_db(self, df: pd.DataFrame):
        # Process dates more efficiently
        rev = pd.to_datetime(df["revdate"], errors="coerce")
        for date_col in DATE_COLUMNS:
            df[date_col] = rev.dt.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare data for database insertion
        items_df = df[ITEM_COLS_CSV].copy()
        items_df.columns = ITEM_COLS_DB

        # Insert items into database
        items_df.to_sql(TABLE_NAME, self.conn, if_exists="append", index=False)
        self.conn.commit()

    def load_db(
        self,
        reader: Iterator[pd.DataFrame],
    ) -> None:
        for df in reader:
            df = df.copy()
            self.load_df_to_db(df)
        self._create_indexes()
        self.cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        print(f"Loaded {self.cur.fetchone()[0]} items into the database")


    def get_from_item_id(self, item_id: int) -> ArticleRecord:
        self.cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE item_id = ?;", (item_id,))
        row = self.cur.fetchone()
        print(row)
        return ArticleRecord.from_row(row)

    def _create_indexes(self):
        for col in ITEM_COLS_DB:
            self.cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_{col}      ON {TABLE_NAME}({col});")
        self.conn.commit()
        self.cur.execute("PRAGMA synchronous = NORMAL;")

if __name__ == "__main__":
    # for initializing the database
    db_management = DbManagement()
    db_management.create_db()
    from settings import OUT_CSV, CHUNK_SIZE
    reader = pd.read_csv(OUT_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    db_management.load_db(reader)