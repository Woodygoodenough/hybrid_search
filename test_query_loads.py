"""Interactive script for testing query loads with evenly distributed survivors."""

from dbManagement import DbManagement
from histo2d import Histo2D
from shared_dataclasses import Predicate
from timer import Timer, SearchMethod, TimedMethodResult, TimedPredicatesResults
from search import Search
from typing import List
import pandas as pd

db = DbManagement()
histo = Histo2D.from_records(db.predicates_search([]))
timer = Timer()
search = Search(timer)
query = "test query"

# Hardcoded single predicate queries (5 queries) - evenly distributed
single_queries = [
    [Predicate(key="revdate", value="2025-02-19 19:08:15", operator=">=")],  # ~15k
    [Predicate(key="revdate", value="2025-02-19 05:16:42", operator=">=")],  # ~45k
    [Predicate(key="revdate", value="2025-02-18 06:42:18", operator=">=")],  # ~75k
    [Predicate(key="revdate", value="2025-01-26 07:03:28", operator=">=")],  # ~105k
    [Predicate(key="token_count", value=75, operator=">=")],  # ~150k
]

# Hardcoded dual predicate queries (5 queries) - evenly distributed
dual_queries = [
    [
        Predicate(key="token_count", value=51, operator=">="),
        Predicate(key="revdate", value="2025-02-19 13:58:20", operator=">="),
    ],  # ~30k
    [
        Predicate(key="token_count", value=101, operator=">="),
        Predicate(key="revdate", value="2025-02-18 20:27:31", operator=">="),
    ],  # ~60k
    [
        Predicate(key="token_count", value=151, operator=">="),
        Predicate(key="revdate", value="2025-02-17 14:19:47", operator=">="),
    ],  # ~90k
    [
        Predicate(key="token_count", value=200, operator=">="),
        Predicate(key="revdate", value="2025-01-24 02:40:15", operator=">="),
    ],  # ~120k
    [
        Predicate(key="token_count", value=250, operator=">="),
        Predicate(key="revdate", value="2025-01-22 10:15:26", operator=">="),
    ],  # ~135k
]

all_queries = single_queries + dual_queries


def time_predicates(predicates: List[Predicate], k: int):
    timed_method_results = []
    for method in SearchMethod:
        with timer.method_context(method):
            for _ in range(1):
                with timer.run():
                    results = search.search(query, predicates, k, method)
            # Create TimedMethodResult after all runs for this method
            timed_method_results.append(
                TimedMethodResult.from_raw_method_runs((timer.method, timer.runs))
            )
    return TimedPredicatesResults(
        predicates=predicates, timed_method_results=timed_method_results
    )


# Test with k=5, k=10, and k=2000
for k in [5, 10, 5000]:
    print(f"\n{'='*80}")
    print(f"TESTING WITH k={k}")
    print(f"{'='*80}\n")

    all_results = []
    for i, predicates in enumerate(all_queries, 1):
        est = histo.estimate_survivors(predicates)
        actual = len(db.predicates_search(predicates))
        pred_str = ", ".join([f"{p.key} {p.operator} {p.value}" for p in predicates])

        print(f"Query {i}: {pred_str}")
        print(f"  Est survivors: {est} | Actual: {actual}")

        timed_predicates_results = time_predicates(predicates, k)
        df = timed_predicates_results.to_df()
        df["query_num"] = i
        df["k"] = k
        df["est_survivors"] = est
        df["actual_survivors"] = actual
        all_results.append(df)

        # Print timing summary
        print(f"  Timing (ms):")
        for _, row in df.iterrows():
            print(
                f"    {row['method']}: total={row['total']:.2f}, db={row['db_search']:.2f}, faiss={row['faiss_search']:.2f}"
            )
        print()

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\nSummary for k={k}:")
        print(
            combined_df[
                [
                    "query_num",
                    "method",
                    "total",
                    "db_search",
                    "faiss_search",
                    "est_survivors",
                ]
            ].to_string()
        )
        print()
