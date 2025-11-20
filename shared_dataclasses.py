from __future__ import annotations
from dataclasses import dataclass
from typing import List
from datetime import datetime
from settings import PREDICATE_COLUMNS, DATE_COLUMNS


@dataclass
class Predicate:
    key: str
    value: str | int | float | List[str | int | float]
    operator: str

    def __post_init__(self):
        if not self._check_valid():
            raise ValueError(f"Invalid predicate: {self}")
        if self.key in DATE_COLUMNS:
            self.value = self._normalize_date_string(self.value)

    def _check_valid(self) -> bool:
        if self.key not in PREDICATE_COLUMNS:
            return False
        if self.operator not in ["=", ">", "<", ">=", "<=", "IN"]:
            return False
        if self.operator == "IN" and not isinstance(self.value, List):
            return False
        return True

    def _normalize_date_string(self, s: str) -> str:
        s = str(s)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                if fmt == "%Y-%m-%d":
                    dt = dt.replace(hour=0, minute=0, second=0)
                elif fmt == "%Y-%m-%d %H:%M":
                    dt = dt.replace(second=0)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        raise ValueError(
            f"Unsupported date format for predicate value '{s}', possible formats: %Y-%m-%d %H:%M:%S, %Y-%m-%d %H:%M, %Y-%m-%d"
        )

    def __str__(self) -> str:
        return f"{self.key} {self.operator} {self.value}"
