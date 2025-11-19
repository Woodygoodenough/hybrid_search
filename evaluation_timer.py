# %%
from timer import Timer, SearchMethod, TimedMethodResult, TimedPredicatesResults
from search import Search
from shared_dataclasses import Predicate
from typing import List
from dbManagement import DbManagement
import pandas as pd

timer = Timer()
db = DbManagement()
search = Search(timer, db)
query = "machine learning"
k_values = [5, 10, 50, 100, 500, 1000, 2500, 5000, 7000, 10000]
single_queries = [
    [Predicate(key="revdate", value="2025-02-19 19:08:15", operator=">=")],  # ~15k
    [Predicate(key="revdate", value="2025-02-19 12:33:47", operator=">=")],  # ~20k
    [Predicate(key="revdate", value="2025-02-19 08:51:23", operator=">=")],  # ~25k
    [Predicate(key="revdate", value="2025-02-19 01:27:56", operator=">=")],  # ~35k
    [Predicate(key="revdate", value="2025-02-18 22:14:38", operator=">=")],  # ~40k
    [Predicate(key="revdate", value="2025-02-19 05:16:42", operator=">=")],  # ~45k
    [Predicate(key="revdate", value="2025-02-18 18:49:12", operator=">=")],  # ~50k
    [Predicate(key="revdate", value="2025-02-18 15:37:29", operator=">=")],  # ~55k
    [Predicate(key="revdate", value="2025-02-18 11:24:51", operator=">=")],  # ~65k
    [Predicate(key="revdate", value="2025-02-18 09:03:17", operator=">=")],  # ~70k
    [Predicate(key="revdate", value="2025-02-18 06:42:18", operator=">=")],  # ~75k
    [Predicate(key="revdate", value="2025-02-17 23:58:44", operator=">=")],  # ~80k
    [Predicate(key="revdate", value="2025-02-17 19:41:33", operator=">=")],  # ~85k
    [Predicate(key="revdate", value="2025-02-17 16:28:07", operator=">=")],  # ~95k
    [Predicate(key="revdate", value="2025-02-17 12:15:52", operator=">=")],  # ~100k
    [Predicate(key="revdate", value="2025-01-26 07:03:28", operator=">=")],  # ~105k
    [Predicate(key="revdate", value="2025-01-25 14:22:19", operator=">=")],  # ~110k
    [Predicate(key="revdate", value="2025-01-25 08:47:35", operator=">=")],  # ~115k
    [Predicate(key="revdate", value="2025-01-24 18:33:21", operator=">=")],  # ~125k
    [Predicate(key="revdate", value="2025-01-24 11:56:48", operator=">=")],  # ~130k
    [Predicate(key="revdate", value="2025-01-23 20:14:07", operator=">=")],  # ~140k
    [Predicate(key="revdate", value="2025-01-23 13:42:55", operator=">=")],  # ~145k
    [Predicate(key="token_count", value=75, operator=">=")],  # ~150k
]

dual_queries = [
    [
        Predicate(key="token_count", value=51, operator=">="),
        Predicate(key="revdate", value="2025-02-19 13:58:20", operator=">="),
    ],  # ~30k
    [
        Predicate(key="token_count", value=45, operator=">="),
        Predicate(key="revdate", value="2025-02-19 10:27:14", operator=">="),
    ],  # ~22k
    [
        Predicate(key="token_count", value=48, operator=">="),
        Predicate(key="revdate", value="2025-02-19 07:41:33", operator=">="),
    ],  # ~27k
    [
        Predicate(key="token_count", value=55, operator=">="),
        Predicate(key="revdate", value="2025-02-19 03:19:48", operator=">="),
    ],  # ~32k
    [
        Predicate(key="token_count", value=60, operator=">="),
        Predicate(key="revdate", value="2025-02-18 23:52:16", operator=">="),
    ],  # ~37k
    [
        Predicate(key="token_count", value=65, operator=">="),
        Predicate(key="revdate", value="2025-02-18 20:11:42", operator=">="),
    ],  # ~42k
    [
        Predicate(key="token_count", value=70, operator=">="),
        Predicate(key="revdate", value="2025-02-18 16:38:27", operator=">="),
    ],  # ~47k
    [
        Predicate(key="token_count", value=85, operator=">="),
        Predicate(key="revdate", value="2025-02-18 13:04:59", operator=">="),
    ],  # ~52k
    [
        Predicate(key="token_count", value=90, operator=">="),
        Predicate(key="revdate", value="2025-02-18 09:28:15", operator=">="),
    ],  # ~57k
    [
        Predicate(key="token_count", value=101, operator=">="),
        Predicate(key="revdate", value="2025-02-18 20:27:31", operator=">="),
    ],  # ~60k
    [
        Predicate(key="token_count", value=95, operator=">="),
        Predicate(key="revdate", value="2025-02-18 05:17:43", operator=">="),
    ],  # ~62k
    [
        Predicate(key="token_count", value=110, operator=">="),
        Predicate(key="revdate", value="2025-02-18 01:49:28", operator=">="),
    ],  # ~67k
    [
        Predicate(key="token_count", value=115, operator=">="),
        Predicate(key="revdate", value="2025-02-17 22:06:14", operator=">="),
    ],  # ~72k
    [
        Predicate(key="token_count", value=120, operator=">="),
        Predicate(key="revdate", value="2025-02-17 18:33:51", operator=">="),
    ],  # ~77k
    [
        Predicate(key="token_count", value=130, operator=">="),
        Predicate(key="revdate", value="2025-02-17 15:11:37", operator=">="),
    ],  # ~82k
    [
        Predicate(key="token_count", value=140, operator=">="),
        Predicate(key="revdate", value="2025-02-17 11:48:22", operator=">="),
    ],  # ~87k
    [
        Predicate(key="token_count", value=151, operator=">="),
        Predicate(key="revdate", value="2025-02-17 14:19:47", operator=">="),
    ],  # ~90k
    [
        Predicate(key="token_count", value=145, operator=">="),
        Predicate(key="revdate", value="2025-02-17 08:25:09", operator=">="),
    ],  # ~92k
    [
        Predicate(key="token_count", value=160, operator=">="),
        Predicate(key="revdate", value="2025-02-17 04:52:46", operator=">="),
    ],  # ~97k
    [
        Predicate(key="token_count", value=170, operator=">="),
        Predicate(key="revdate", value="2025-02-17 01:19:33", operator=">="),
    ],  # ~102k
    [
        Predicate(key="token_count", value=180, operator=">="),
        Predicate(key="revdate", value="2025-01-26 11:47:18", operator=">="),
    ],  # ~107k
    [
        Predicate(key="token_count", value=190, operator=">="),
        Predicate(key="revdate", value="2025-01-26 04:28:55", operator=">="),
    ],  # ~112k
    [
        Predicate(key="token_count", value=200, operator=">="),
        Predicate(key="revdate", value="2025-01-24 02:40:15", operator=">="),
    ],  # ~120k
    [
        Predicate(key="token_count", value=210, operator=">="),
        Predicate(key="revdate", value="2025-01-24 19:16:42", operator=">="),
    ],  # ~122k
    [
        Predicate(key="token_count", value=220, operator=">="),
        Predicate(key="revdate", value="2025-01-24 15:53:28", operator=">="),
    ],  # ~127k
    [
        Predicate(key="token_count", value=230, operator=">="),
        Predicate(key="revdate", value="2025-01-23 12:30:14", operator=">="),
    ],  # ~132k
    [
        Predicate(key="token_count", value=250, operator=">="),
        Predicate(key="revdate", value="2025-01-22 10:15:26", operator=">="),
    ],  # ~135k
    [
        Predicate(key="token_count", value=240, operator=">="),
        Predicate(key="revdate", value="2025-01-23 08:47:52", operator=">="),
    ],  # ~137k
    [
        Predicate(key="token_count", value=260, operator=">="),
        Predicate(key="revdate", value="2025-01-22 06:24:38", operator=">="),
    ],  # ~142k
    [
        Predicate(key="token_count", value=270, operator=">="),
        Predicate(key="revdate", value="2025-01-22 03:01:24", operator=">="),
    ],  # ~147k
]


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
    num_survivors = len(db.predicates_search(predicates))
    return TimedPredicatesResults(
        predicates=predicates,
        timed_method_results=timed_method_results,
        k=k,
        num_survivors=num_survivors,
    )


df_all = pd.DataFrame()
for predicates in single_queries + dual_queries:
    for k in k_values:
        timed_predicates_results = time_predicates(predicates, k)
        df_all = pd.concat([df_all, timed_predicates_results.to_df()])
df_all.to_csv("timed_results.csv", index=False)

# %%
df_all
# %%
