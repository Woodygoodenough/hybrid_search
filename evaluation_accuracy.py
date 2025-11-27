# Accuracy evaluation for hybrid search methods
# Uses same test cases as evaluation_timer.py but evaluates against brute force ground truth

from search import Search
from shared_dataclasses import Predicate
from timer import SearchMethod
import numpy as np
import pandas as pd
import json

# Same setup as evaluation_timer.py
query = "machine learning"
k_values = [5, 10, 50, 100, 200, 500, 1000, 2500, 5000, 6000, 7000, 10000]

# Single predicate queries (same as evaluation_timer.py)
single_queries = [
    [Predicate(key="token_count", value=400, operator=">=")],  # ~300
    [Predicate(key="token_count", value=350, operator=">=")],  # ~1000
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

# Dual predicate queries (same as evaluation_timer.py)
dual_queries = [
    [
        Predicate(key="token_count", value=450, operator=">="),
        Predicate(key="revdate", value="2025-02-19 23:47:33", operator=">="),
    ],  # ~100
    [
        Predicate(key="token_count", value=380, operator=">="),
        Predicate(key="revdate", value="2025-02-19 21:15:28", operator=">="),
    ],  # ~500
    [
        Predicate(key="token_count", value=300, operator=">="),
        Predicate(key="revdate", value="2025-02-19 18:42:17", operator=">="),
    ],  # ~5000
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


# Accuracy metric functions
def calc_recall(predicted_ids, truth_ids):
    """Recall: what fraction of ground truth results did we find?"""
    if len(truth_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0
    matches = len(predicted_ids.intersection(truth_ids))
    return matches / len(truth_ids)


def calc_precision(predicted_ids, truth_ids):
    """Precision: what fraction of our results are correct?"""
    if len(predicted_ids) == 0:
        return 1.0 if len(truth_ids) == 0 else 0.0
    matches = len(predicted_ids.intersection(truth_ids))
    return matches / len(predicted_ids)


def calc_ndcg(pred_results, truth_results, k):
    """NDCG: how good is our ranking compared to ground truth?"""
    truth_scores = {r.record.item_id: r.similarity for r in truth_results}
    
    # DCG for predictions
    dcg = 0.0
    for i, result in enumerate(pred_results[:k]):
        score = truth_scores.get(result.record.item_id, 0.0)
        dcg += score / np.log2(i + 2)
    
    # ideal DCG
    idcg = 0.0
    for i, result in enumerate(truth_results[:k]):
        idcg += result.similarity / np.log2(i + 2)
    
    if idcg == 0:
        return 1.0 if dcg == 0 else 0.0
    
    return dcg / idcg


def evaluate_accuracy(search_engine, predicates, k):
    """Test all methods against brute force ground truth"""
    
    # get ground truth using brute force
    gt = search_engine.search(query, predicates, k, SearchMethod.BRUTE_FORCE)
    
    if len(gt.results) == 0:
        return None
    
    gt_ids = set(r.record.item_id for r in gt.results[:k])
    
    # test each search method
    methods = [
        SearchMethod.BASE_PRE_SEARCH,
        SearchMethod.BASE_POS_SEARCH,
        SearchMethod.ADAP_PRE_SEARCH,
        SearchMethod.ADAP_POS_SEARCH,
    ]
    
    results = {}
    for method in methods:
        method_results = search_engine.search(query, predicates, k, method)
        method_ids = set(r.record.item_id for r in method_results.results)
        
        results[method.name] = {
            'recall': calc_recall(method_ids, gt_ids),
            'precision': calc_precision(method_ids, gt_ids),
            'ndcg': calc_ndcg(method_results.results, gt.results, k),
            'num_results': len(method_results.results)
        }
    
    return results


# main evaluation
print("="*80)
print("ACCURACY EVALUATION")
print("="*80)
print(f"Total test cases: {len(single_queries + dual_queries) * len(k_values)}")
print("Using same test cases as evaluation_timer.py")
print("This will take several hours due to brute force computation...\n")

accuracy_data = []
test_num = 0
total_tests = len(single_queries + dual_queries) * len(k_values)

with Search() as search_engine:
    for predicates in single_queries + dual_queries:
        for k in k_values:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Predicates: {predicates}, k={k}")
            
            # run accuracy evaluation
            print("  Computing ground truth (brute force)...")
            accuracy = evaluate_accuracy(search_engine, predicates, k)
            
            if accuracy is None:
                print("  WARNING: No results found, skipping")
                continue
            
            # save data
            accuracy_data.append({
                'predicates': str(predicates),
                'k': k,
                **{f"{method}_recall": metrics['recall'] 
                   for method, metrics in accuracy.items()},
                **{f"{method}_precision": metrics['precision'] 
                   for method, metrics in accuracy.items()},
                **{f"{method}_ndcg": metrics['ndcg'] 
                   for method, metrics in accuracy.items()},
                **{f"{method}_num_results": metrics['num_results'] 
                   for method, metrics in accuracy.items()},
            })
            
            # print summary
            print("  Results:")
            for method, metrics in accuracy.items():
                method_short = method.replace('_SEARCH', '')
                print(f"    {method_short:<15} Recall: {metrics['recall']:.4f}  "
                      f"Precision: {metrics['precision']:.4f}  NDCG: {metrics['ndcg']:.4f}")

# save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

df = pd.DataFrame(accuracy_data)
df.to_csv("accuracy_results.csv", index=False)
print("✓ Saved to: accuracy_results.csv")

with open("accuracy_results.json", "w") as f:
    json.dump(accuracy_data, f, indent=2)
print("✓ Saved to: accuracy_results.json")

# print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

methods = ['BASE_PRE_SEARCH', 'BASE_POS_SEARCH', 'ADAP_PRE_SEARCH', 'ADAP_POS_SEARCH']
print(f"\n{'Method':<20} {'Avg Recall':<12} {'Avg Precision':<12} {'Avg NDCG':<12}")
print("-" * 70)

for method in methods:
    avg_recall = df[f"{method}_recall"].mean()
    avg_precision = df[f"{method}_precision"].mean()
    avg_ndcg = df[f"{method}_ndcg"].mean()
    method_short = method.replace('_SEARCH', '')
    print(f"{method_short:<20} {avg_recall:<12.4f} {avg_precision:<12.4f} {avg_ndcg:<12.4f}")

print(f"\n✓ Evaluation complete: {len(accuracy_data)} test cases")