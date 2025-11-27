from __future__ import annotations
from dataclasses import dataclass
import sqlite3
from settings import (
    DB_PATH,
    USECOLS,
    ITEM_COLS_DB,
    ITEM_COLS_CSV,
    PRIMARY_KEY,
    TABLE_NAME,
    DATE_COLUMNS,
    INT_COLUMNS_CSV,
)
import pandas as pd
from typing import Iterator, List, Tuple, Optional
from shared_dataclasses import Predicate

@dataclass
class DBRecord:
    item_id: int
    ext_id: int
    title: str
    url: str
    revdate: str
    token_count: int
    entity: str

    @classmethod
    def from_row(cls, row: Tuple[int, int, str, str, str, int, str]) -> "DBRecord":
        return cls(
            item_id=row[0],
            ext_id=row[1],
            title=row[2],
            url=row[3],
            revdate=row[4],
            token_count=row[5],
            entity=row[6],
        )

    @classmethod
    def get_attrs(cls) -> List[str]:
        return ["item_id", "ext_id", "title", "url", "revdate", "token_count", "entity"]

@dataclass
class DBRecords:
    records: List[DBRecord]
    
    def to_df(self, show_cols: Optional[List[str]] = None) -> pd.DataFrame:
        # we should consider empty records, we still return a dataframe with correct columns, but with empty rows
        attrs = DBRecord.get_attrs()
        if len(self.records) == 0:
            return pd.DataFrame(columns=attrs)
        df = pd.DataFrame([record.__dict__ for record in self.records])
        if show_cols:
            df = df.loc[:, show_cols]
        return df
    
    def to_dict(self):
        # return a dict with item_id as key and record as value
        return {record.item_id: record for record in self.records}
    
    def __len__(self) -> int:
        return len(self.records)
        
    def __iter__(self) -> Iterator[DBRecord]:
        return iter(self.records)

    # define index operator
    def __getitem__(self, item_id: int) -> DBRecord:
        # preserve KeyError
        return self.to_dict()[item_id]


class DbManagement:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cur = self.conn.cursor()
        self.histogram_2d = None  # Will store 2D histogram for date vs token_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close database connection."""
        try:
            if hasattr(self, 'cur') and self.cur:
                self.cur.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception:
            pass

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
        sql_prep = f"CREATE TABLE {TABLE_NAME} ("
        for col in ITEM_COLS_DB:
            if col in INT_COLUMNS_CSV:
                sql_prep += f"{col} INTEGER, "
            else:
                sql_prep += f"{col} TEXT, "
        sql_prep += f"PRIMARY KEY ({PRIMARY_KEY}))"
        self.cur.execute(sql_prep)
        self.conn.commit()

    def _load_df_to_db(self, df: pd.DataFrame):
        # Process dates more efficiently
        rev = pd.to_datetime(df["revdate"], errors="coerce")
        for date_col in DATE_COLUMNS:
            df[date_col] = rev.dt.strftime("%Y-%m-%d %H:%M:%S")
        
        for int_col in INT_COLUMNS_CSV:
            df[int_col] = df[int_col].astype(int)
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
            self._load_df_to_db(df)
        self._create_indexes()
        self.cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        print(f"Loaded {self.cur.fetchone()[0]} items into the database")

    def get_from_item_id(self, item_id: int) -> DBRecord:
        self.cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE item_id = ?;", (item_id,))
        row = self.cur.fetchone()
        if row is None:
            raise ValueError(f"No record found for item_id: {item_id}")
        return DBRecord.from_row(row)

    def predicates_search(self, predicates: List[Predicate]) -> DBRecords:        
        if predicates:
            sql_prep, val_prep_list = DbManagement.predicates_sql_prep(predicates)
        else:
            sql_prep = f"SELECT * FROM {TABLE_NAME}"
            val_prep_list = []
        self.cur.execute(sql_prep, tuple(val_prep_list))
        rows = self.cur.fetchall()
        return DBRecords(records=[DBRecord.from_row(row) for row in rows])

    @staticmethod
    def predicates_sql_prep(predicates: List[Predicate]) -> Tuple[str, List[str | int | float | List[str | int | float]]]:
        sql_prep = f"SELECT * FROM {TABLE_NAME} WHERE "
        val_prep_list = []
        for predicate in predicates:
            if predicate.operator == "IN":
                normalized_list = predicate.value
                str_prep = f"{predicate.key} IN ({','.join(['?' for _ in normalized_list])})"
                val_prep_list.extend(normalized_list)  # type: ignore[arg-type]
            else:
                str_prep = f"{predicate.key} {predicate.operator} ?"
                val_prep_list.append(predicate.value)
            sql_prep += str_prep + " AND "
        # Remove the trailing " AND " from the end
        sql_prep = sql_prep[:-5]
        return sql_prep, val_prep_list

    def _create_indexes(self):
        for col in ITEM_COLS_DB:
            self.cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_{col}      ON {TABLE_NAME}({col});"
            )
        self.conn.commit()
        self.cur.execute("PRAGMA synchronous = NORMAL;")


if __name__ == "__main__":
    # for initializing the database
    db_management = DbManagement()
    db_management.create_db()
    from settings import CHUNK_SIZE

    reader = pd.read_csv("wikipedia_sample_150k_with_ids.csv", usecols=USECOLS, chunksize=CHUNK_SIZE)
    db_management.load_db(reader)
