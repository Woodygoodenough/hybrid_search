from __future__ import annotations
from dataclasses import dataclass
import time
from contextlib import contextmanager
from typing import List, Tuple, Optional
from statistics import mean
from enum import Enum
import pandas as pd
from shared_dataclasses import Predicate

# named tuples for type hints
from typing import NamedTuple

TimedSection = NamedTuple(
    "TimedSection", [("section_name", str), ("execution_time", float)]
)


# your enums
class SearchMethod(Enum):
    BASE_PRE_SEARCH = 1
    ADAP_PRE_SEARCH = 2
    BASE_POS_SEARCH = 3
    ADAP_POS_SEARCH = 4


class Section(Enum):
    DB_SEARCH = 1
    HISTO_FILTER = 2
    FAISS_SEARCH = 3
    UD_PARAMS = 4
    INTERSECT = 5
    RESIDUAL = 6
    FINALIZE = 7
    TOTAL = 8


# typed aliases
TimedSection = Tuple[Section, float]  # (section_enum, elapsed_seconds)
TimedRun = List[TimedSection]  # one run = list of sections
TimedRuns = List[TimedRun]  # many runs for a method


class MethodResults:
    def __init__(self):
        self.method: Optional[SearchMethod] = None
        self.runs: TimedRuns = []  # finished runs for the current method
        self._current_run: Optional[TimedRun] = None


class Timer:
    """Efficient timer for tracking different parts of search methods."""

    def __init__(self):
        self.method: Optional[SearchMethod] = None
        self.runs: TimedRuns = []  # finished runs for the current method
        self._current_run: Optional[TimedRun] = None

    @contextmanager
    def method_context(self, method: SearchMethod):
        """
        Enter a method context which can contain multiple runs.
        We only use it when testing timing.
        """
        # set method-level state
        self.method = method
        self.runs = []
        try:
            yield
        finally:
            self._current_run = None

    @contextmanager
    def run(self):
        """Start a single run inside the current method."""
        if self.method is None:
            raise RuntimeError("Must enter method_context before starting runs.")
        self._current_run = []
        try:
            yield
        finally:
            # finalize run and append to runs
            assert self._current_run is not None
            self.runs.append(self._current_run)
            self._current_run = None

    @contextmanager
    def section(self, name: Section):
        """Time a section of code - low overhead.
        We only use it when testing timing.
        """
        if self._current_run is None:
            # No active run: act as a no-op context so normal runs can use `with` safely
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            # record as enum + elapsed
            self._current_run.append((name, elapsed))

    def reset(self):
        self.runs = []
        self._current_run = None
        self.method = None


RawMethodRuns = Tuple[SearchMethod, TimedRuns]


@dataclass
class TimedMethodResult:
    method: SearchMethod
    total: float
    db_search: float
    histo_filter: float
    faiss_search: float
    ud_params: float
    intersect: float
    residual: float
    finalize: float
    iterations: int = None
    num_runs: int = None

    @classmethod
    def from_raw_method_runs(cls, raw_method_runs: RawMethodRuns) -> TimedMethodResult:
        method, runs = raw_method_runs
        # aggregate times across all runs (allow multiple same sections per run)
        section_times = {section: 0.0 for section in Section}
        for run in runs:
            for section, time in run:
                section_times[section] += time
        iterations_per_run = [
            sum(1 for s, _ in run if s == Section.FAISS_SEARCH) for run in runs
        ]
        # check if all iterations are the same
        if not all(
            iterations == iterations_per_run[0] for iterations in iterations_per_run
        ):
            raise ValueError("All iterations must be the same")
        iterations = iterations_per_run[0]
        num_runs = len(runs)
        # compute residual as TOTAL minus all explicitly measured non-residual sections
        measured_non_residual = (
            section_times[Section.DB_SEARCH]
            + section_times[Section.HISTO_FILTER]
            + section_times[Section.FAISS_SEARCH]
            + section_times[Section.UD_PARAMS]
            + section_times[Section.INTERSECT]
            + section_times[Section.FINALIZE]
        )
        total_time = section_times[Section.TOTAL]
        residual_time = max(0.0, total_time - measured_non_residual)
        return cls(
            method=method,
            total=total_time,
            db_search=section_times[Section.DB_SEARCH],
            histo_filter=section_times[Section.HISTO_FILTER],
            faiss_search=section_times[Section.FAISS_SEARCH],
            ud_params=section_times[Section.UD_PARAMS],
            intersect=section_times[Section.INTERSECT],
            residual=residual_time,
            finalize=section_times[Section.FINALIZE],
            iterations=iterations,
            num_runs=num_runs,
        )

    def to_ave_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "method": [self.method],
                "total": [self.total / self.num_runs],
                "iterations": [self.iterations],
                "db_search": [self.db_search / self.num_runs],
                "histo_filter": [self.histo_filter / self.num_runs],
                "faiss_search": [self.faiss_search / self.num_runs],
                "ud_params": [self.ud_params / self.num_runs],
                "intersect": [self.intersect / self.num_runs],
                "residual": [self.residual / self.num_runs],
                "finalize": [self.finalize / self.num_runs],
            }
        )


@dataclass
class TimedPredicatesResults:
    predicates: List[Predicate]
    timed_method_results: List[TimedMethodResult]

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        # we know do not add predicates to the dataframe. maybe later with other representations.
        for timed_method_result in self.timed_method_results:
            df = pd.concat([df, timed_method_result.to_ave_df()])
        return df
