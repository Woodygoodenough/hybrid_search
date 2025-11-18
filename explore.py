# %%
from timer import Timer, SearchMethod, TimedMethodResult, TimedPredicatesResults
from search import Search
from shared_dataclasses import Predicate
from typing import List

timer = Timer()
search = Search(timer)

query = "test query"
predicates = [Predicate(key="revdate", value="2025-01-15", operator=">=")]
k = 1


def time_predicates(predicates: List[Predicate], k: int):
    timed_method_results = []
    for method in SearchMethod:
        with timer.method_context(method):
            for _ in range(k):
                with timer.run():
                    results = search.search(query, predicates, k, method)
                timed_method_results.append(
                    TimedMethodResult.from_raw_method_runs((timer.method, timer.runs))
                )
                timer.reset()
    return TimedPredicatesResults(
        predicates=predicates, timed_method_results=timed_method_results
    )


timed_predicates_results = time_predicates(predicates, k)
timed_predicates_results.to_df()
