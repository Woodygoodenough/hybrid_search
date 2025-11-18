"""Interactive script for testing query loads with evenly distributed survivors."""

from dbManagement import DbManagement
from histo2d import Histo2D
from shared_dataclasses import Predicate

db = DbManagement()
histo = Histo2D.from_records(db.predicates_search([]))

# Hardcoded single predicate queries (5 queries)
single_queries = [
    [Predicate(key="revdate", value="2025-02-19 19:10:19", operator=">=")],
    [Predicate(key="token_count", value=6763, operator=">=")],
    [Predicate(key="revdate", value="2025-02-19 13:58:20", operator=">=")],
    [Predicate(key="token_count", value=3908, operator=">=")],
    [Predicate(key="revdate", value="2025-02-19 05:14:37", operator=">=")],
]

# Hardcoded dual predicate queries (5 queries)
dual_queries = [
    [
        Predicate(key="token_count", value=2575, operator=">="),
        Predicate(key="revdate", value="2025-02-19 05:14:37", operator=">="),
    ],
    [
        Predicate(key="token_count", value=1786, operator=">="),
        Predicate(key="revdate", value="2025-02-19 05:14:37", operator=">="),
    ],
    [
        Predicate(key="token_count", value=2575, operator=">="),
        Predicate(key="revdate", value="2025-02-18 20:27:31", operator=">="),
    ],
    [
        Predicate(key="token_count", value=1289, operator=">="),
        Predicate(key="revdate", value="2025-02-19 05:14:37", operator=">="),
    ],
    [
        Predicate(key="token_count", value=2575, operator=">="),
        Predicate(key="revdate", value="2025-02-18 06:44:23", operator=">="),
    ],
]

# Print results
print("SINGLE PREDICATE QUERIES:")
for i, predicates in enumerate(single_queries, 1):
    est = histo.estimate_survivors(predicates)
    actual = len(db.predicates_search(predicates))
    pred_str = ", ".join([f"{p.key} {p.operator} {p.value}" for p in predicates])
    print(f"{i}. {pred_str} | Est: {est} | Actual: {actual}")

print("\nDUAL PREDICATE QUERIES:")
for i, predicates in enumerate(dual_queries, 1):
    est = histo.estimate_survivors(predicates)
    actual = len(db.predicates_search(predicates))
    pred_str = ", ".join([f"{p.key} {p.operator} {p.value}" for p in predicates])
    print(f"{i}. {pred_str} | Est: {est} | Actual: {actual}")
