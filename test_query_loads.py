"""Interactive script for testing query loads with evenly distributed survivors."""

from dbManagement import DbManagement
from histo2d import Histo2D
from shared_dataclasses import Predicate

db = DbManagement()
histo = Histo2D.from_records(db.predicates_search([]))

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
