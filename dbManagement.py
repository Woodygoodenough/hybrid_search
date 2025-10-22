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
)
import pandas as pd
from typing import Iterator, List, Tuple, Dict
from settings import PREDICATE_COLUMNS
from datetime import datetime


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
        return cls(
            item_id=row[0],
            ext_id=row[1],
            title=row[2],
            url=row[3],
            revdate=row[4],
            token_count=row[5],
            entity=row[6],
        )


@dataclass
class Predicate:
    key: str
    value: str
    operator: str

    def __post_init__(self):
        if not self.check_valid():
            raise ValueError(f"Invalid predicate: {self}")

    def check_valid(self) -> bool:
        if self.key not in PREDICATE_COLUMNS:
            return False
        if self.operator not in ["=", ">", "<", ">=", "<="]:
            return False
        return True

    def to_where_sql(self) -> Tuple[str, str]:
        str_prep = f"WHERE {self.key} {self.operator} ?"
        val_prep = self.value
        return str_prep, val_prep


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
        sql_prep = f"CREATE TABLE {TABLE_NAME} ("
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
        if row is None:
            raise ValueError(f"No record found for item_id: {item_id}")
        return ArticleRecord.from_row(row)

    def query_db(self, query: str) -> List[ArticleRecord]:
        return [
            ArticleRecord.from_row(row) for row in self.cur.execute(query).fetchall()
        ]

    def predicates_search(self, predicates: List[Predicate]) -> List[ArticleRecord]:
        sql_prep = f"SELECT * FROM {TABLE_NAME} "
        val_prep = []
        for predicate in predicates:
            str_prep, val_prep = predicate.to_where_sql()
            sql_prep += str_prep
            val_prep.append(val_prep)
        self.cur.execute(sql_prep, tuple(val_prep))
        rows = self.cur.fetchall()
        return [ArticleRecord.from_row(row) for row in rows]

    def describe_columns(self, cols: List[str]) -> Dict[str, Dict[str, float]]:
        # for each col, describe the min, max, median, 1st quartile, 3rd quartile
        # SQLite does not provide MEDIAN/QUANTILE; compute from ordered values.
        desc_dict = {
            col: {"min": None, "max": None, "median": None, "q1": None, "q3": None}
            for col in cols
        }
        for col in cols:
            if col in DATE_COLUMNS:
                # Use Unix seconds for ordering and quantiles
                self.cur.execute(
                    f"SELECT strftime('%s', {col}) AS ts "
                    f"FROM {TABLE_NAME} "
                    f"WHERE {col} IS NOT NULL "
                    f"ORDER BY ts ASC;"
                )
                ts_values = [int(r[0]) for r in self.cur.fetchall()]
                if not ts_values:
                    continue
                desc_dict[col]["min"] = datetime.fromtimestamp(ts_values[0]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                desc_dict[col]["max"] = datetime.fromtimestamp(ts_values[-1]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                # quantiles via interpolation on numeric timestamps
                median_ts = int(self._percentile(ts_values, 0.5))
                q1_ts = int(self._percentile(ts_values, 0.25))
                q3_ts = int(self._percentile(ts_values, 0.75))
                desc_dict[col]["median"] = datetime.fromtimestamp(median_ts).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                desc_dict[col]["q1"] = datetime.fromtimestamp(q1_ts).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                desc_dict[col]["q3"] = datetime.fromtimestamp(q3_ts).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                # numeric-like columns
                self.cur.execute(
                    f"SELECT CAST({col} AS REAL) AS v "
                    f"FROM {TABLE_NAME} "
                    f"WHERE {col} IS NOT NULL "
                    f"ORDER BY v ASC;"
                )
                values = [row[0] for row in self.cur.fetchall() if row[0] is not None]
                if not values:
                    continue
                desc_dict[col]["min"] = float(values[0])
                desc_dict[col]["max"] = float(values[-1])
                desc_dict[col]["median"] = float(self._percentile(values, 0.5))
                desc_dict[col]["q1"] = float(self._percentile(values, 0.25))
                desc_dict[col]["q3"] = float(self._percentile(values, 0.75))
        return desc_dict

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Compute the p-quantile using linear interpolation on pre-sorted values."""
        n = len(sorted_values)
        if n == 0:
            return float("nan")
        if p <= 0:
            return float(sorted_values[0])
        if p >= 1:
            return float(sorted_values[-1])
        rank = p * (n - 1)
        low_index = int(rank)
        high_index = low_index + 1
        fraction = rank - low_index
        if high_index >= n:
            return float(sorted_values[low_index])
        low_val = sorted_values[low_index]
        high_val = sorted_values[high_index]
        return float(low_val + (high_val - low_val) * fraction)

    def desc_to_df(self, desc_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(desc_dict)
        return df

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
    from settings import OUT_CSV, CHUNK_SIZE

    reader = pd.read_csv(OUT_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    db_management.load_db(reader)
