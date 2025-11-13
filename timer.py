from __future__ import annotations
import time
from contextlib import contextmanager
from typing import List, Tuple, Optional

from enum import Enum

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
            # we write the method results to memory for later analysis
            # to be implemented
            # to maintain
            pass

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
            elapsed = time.perf_counter() - start
            # record as enum + elapsed
            self._current_run.append((name, elapsed))

    # convenience helper to pretty-print results
    def summarize(self) -> str:
        lines = []
        for i, run in enumerate(self.runs, 1):
            total = sum(t for _, t in run)
            lines.append(f"Run {i}: total {total:.4f}s")
            for sec, t in run:
                lines.append(f"  {sec.name}: {t:.4f}s")
        return "\n".join(lines)

    """
    def _aggregate_times(
        self, raw_times: List[TimedSection] = None
    ) -> defaultdict[str, float]:
        # use defaultdict to aggregate times

        if raw_times is None:
            raw_times = self.raw_times
        aggregated = defaultdict(float)
        for section_name, execution_time in raw_times:
            aggregated[section_name] += execution_time
        return aggregated
    """

    def save_run(self):
        """Save current run for averaging."""
        if self.raw_times:
            self.runs.append(self.raw_times.copy())

    def reset(self):
        self.raw_times.clear()

    def clear_runs(self):
        self.runs.clear()


"""
    def get_averaged_times(self) -> defaultdict[str, float]:

        if not self.runs:
            return self._aggregate_times()

        all_aggregated = defaultdict(list)
        for run in self.runs:
            aggregated = self._aggregate_times(run)
            for name, elapsed in aggregated.items():
                if name not in all_aggregated:
                    all_aggregated[name] = []
                all_aggregated[name].append(elapsed)

        return {name: sum(times) / len(times) for name, times in all_aggregated.items()}

    def to_dataframe(self) -> pd.DataFrame:
        aggregated = self.get_averaged_times()

        # Detect method from section names
        method_name, prefix = self._detect_method(aggregated)
        if method_name is None:
            return pd.DataFrame(
                columns=[
                    "method",
                    "total_time",
                    "db_time",
                    "faiss_time",
                    "histo_time",
                    "residual_time",
                ]
            )

        # Get method total time
        method_total_time = aggregated.get(
            f"{method_name}_total", 0.0
        ) or aggregated.get(f"{prefix}total", 0.0)

        # Categorize times
        db_time, faiss_time, histo_time, residual_time, residual_breakdown = (
            self._categorize_times(aggregated, method_name, prefix, method_total_time)
        )

        # Calculate total time
        total_time = (
            method_total_time + histo_time
            if method_total_time > 0
            else db_time + faiss_time + histo_time + residual_time
        )

        # Create dataframe
        MS_PER_SEC = 1000.0
        return pd.DataFrame(
            [
                {
                    "method": method_name,
                    "total_time": round(total_time * MS_PER_SEC, 2),
                    "db_time": round(db_time * MS_PER_SEC, 2),
                    "faiss_time": round(faiss_time * MS_PER_SEC, 2),
                    "histo_time": round(histo_time * MS_PER_SEC, 2),
                    "residual_time": round(residual_time * MS_PER_SEC, 2),
                    "_residual_intersect": round(
                        residual_breakdown["intersect"] * MS_PER_SEC, 2
                    ),
                    "_residual_prepare": round(
                        residual_breakdown["prepare"] * MS_PER_SEC, 2
                    ),
                    "_residual_update": round(
                        residual_breakdown["update_params"] * MS_PER_SEC, 2
                    ),
                    "_residual_opt": round(residual_breakdown["opt"] * MS_PER_SEC, 2),
                    "_residual_finalize": round(
                        residual_breakdown["finalize"] * MS_PER_SEC, 2
                    ),
                    "_residual_other": round(
                        residual_breakdown["other"] * MS_PER_SEC, 2
                    ),
                }
            ]
        )

    def _detect_method(
        self, aggregated: Dict[str, float]
    ) -> tuple[str | None, str | None]:
        if any(name.startswith("pre_search_") for name in aggregated.keys()):
            return "base_pre_search", "pre_search_"
        elif any(name.startswith("adaptive_pre_search_") for name in aggregated.keys()):
            return "adap_pre_search", "adaptive_pre_search_"
        elif any(name.startswith("adaptive_pos_search_") for name in aggregated.keys()):
            has_opt_params = any(
                "opt_params" in name
                for name in aggregated.keys()
                if name.startswith("adaptive_pos_search_")
            )
            return (
                "adap_pos_search" if has_opt_params else "base_pos_search",
                "adaptive_pos_search_",
            )
        return None, None

    def _categorize_times(
        self,
        aggregated: Dict[str, float],
        method_name: str,
        prefix: str,
        method_total_time: float,
    ):
        db_time = faiss_time = histo_time = residual_time = 0.0
        residual_breakdown = {
            "intersect": 0.0,
            "prepare": 0.0,
            "update_params": 0.0,
            "opt": 0.0,
            "finalize": 0.0,
            "other": 0.0,
        }
        is_adaptive = method_name in ["adap_pre_search", "adap_pos_search"]

        for section_name, elapsed in aggregated.items():
            is_total = (
                section_name.endswith("_total")
                or section_name == f"{method_name}_total"
            )
            belongs_to_method = section_name.startswith(prefix)
            is_histogram = section_name == "histogram_estimation"

            if belongs_to_method and not is_total:
                if "db_filter" in section_name:
                    db_time += elapsed
                elif "faiss" in section_name:
                    faiss_time += elapsed
                elif "intersect" in section_name:
                    residual_time += elapsed
                    residual_breakdown["intersect"] += elapsed
                elif (
                    "prepare_predicates" in section_name
                    or "prepare_item_ids" in section_name
                ):
                    residual_time += elapsed
                    residual_breakdown["prepare"] += elapsed
                elif "update_params" in section_name or "update_nprobe" in section_name:
                    residual_time += elapsed
                    residual_breakdown["update_params"] += elapsed
                elif "opt_params" in section_name or "opt_nprobe" in section_name:
                    residual_time += elapsed
                    residual_breakdown["opt"] += elapsed
                elif "finalize" in section_name:
                    residual_time += elapsed
                    residual_breakdown["finalize"] += elapsed
                else:
                    residual_time += elapsed
                    if "db_filter" not in section_name and "faiss" not in section_name:
                        residual_breakdown["other"] += elapsed

            if is_histogram and is_adaptive:
                histo_time += elapsed

        # Adjust residual if we have method_total_time
        if method_total_time > 0:
            calculated_total = db_time + faiss_time + residual_time
            residual_time += method_total_time - calculated_total

        return db_time, faiss_time, histo_time, residual_time, residual_breakdown




# Global timer instance
_timer = Timer()


def time_section(name: str):
    return _timer.section(name)


def reset_timings():
    _timer.reset()


def save_run():
    _timer.save_run()


def clear_runs():
    _timer.clear_runs()


def benchmark(
    query: str,
    predicates: List[Any],
    num_runs: int = 10,
    warmup_runs: int = 1,
    k: int = 10,
    print_results: bool = True,
) -> pd.DataFrame:
    from search import Search

    if print_results:
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Predicates: {predicates}")
        print(f"Runs: {num_runs} (warmup: {warmup_runs})")
        print("=" * 70)

    search = Search()
    embedding_query = search.embedder.encode_query(query)
    clear_runs()

    # Test all four methods
    methods = [
        (
            "base_pre_search",
            lambda: search.base_pre_search(embedding_query, predicates, k),
            False,
        ),
        (
            "adap_pre_search",
            lambda: search.adap_pre_search(embedding_query, predicates, k),
            True,
        ),
        (
            "base_pos_search",
            lambda: search.base_pos_search(embedding_query, predicates, k),
            False,
        ),
        (
            "adap_pos_search",
            lambda: search.adap_pos_search(
                embedding_query, predicates, k, est_survivors
            ),
            True,
        ),
    ]

    est_survivors = None
    for method_name, method_func, needs_histo in methods:
        # Warmup
        for _ in range(warmup_runs):
            if needs_histo and est_survivors is None:
                est_survivors = search.histo.estimate_survivors(predicates)
            method_func()

        # Timed runs
        for _ in range(num_runs):
            reset_timings()
            if needs_histo:
                with time_section("histogram_estimation"):
                    est_survivors = search.histo.estimate_survivors(predicates)
            method_func()
            save_run()

    all_dfs = []
    num_methods = len(methods)
    for i, (method_name, _, _) in enumerate(methods):
        start_idx = i * num_runs
        end_idx = (i + 1) * num_runs
        method_runs = _timer.runs[start_idx:end_idx]
        if method_runs:
            temp_timer = Timer()
            temp_timer.runs = method_runs
            df = temp_timer.to_dataframe(use_averaged=True)
            if len(df) > 0:
                all_dfs.append(df)

    search.close()

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values("method").reset_index(drop=True)

        if print_results:
            print("\n" + "=" * 70)
            print(f"TIMING BREAKDOWN (Averaged over {num_runs} runs, times in ms)")
            print("=" * 70)
            main_cols = [
                "method",
                "total_time",
                "db_time",
                "faiss_time",
                "histo_time",
                "residual_time",
            ]
            print(df[main_cols].to_string(index=False))

            print("\n" + "=" * 70)
            print("RESIDUAL TIME BREAKDOWN (times in ms)")
            print("=" * 70)
            residual_cols = ["method"] + [
                col for col in df.columns if col.startswith("_residual_")
            ]
            residual_df = df[residual_cols].copy()
            residual_df.columns = [
                col.replace("_residual_", "") if col.startswith("_residual_") else col
                for col in residual_df.columns
            ]
            print(residual_df.to_string(index=False))

        return df
    return pd.DataFrame()
"""
