"""
Example usage of lr_based_adap_search

This script demonstrates how to use the new lr_based_adap_search function
that automatically selects between PRE and POS search based on the trained LR model.
"""

from search import Search
from shared_dataclasses import Predicate
from timer import Timer


def example_lr_based_search():
    """Example of using LR-based adaptive search"""

    # Create search instance (will automatically load lr_model.pkl if it exists)
    timer = Timer()
    search = Search(timer=timer)

    # Example query
    query = "machine learning algorithms"

    # Example predicates
    predicates = [
        Predicate(key="token_count", value=400, operator=">="),
    ]

    # Number of results to retrieve
    k = 10

    print("=" * 80)
    print("LR-BASED ADAPTIVE SEARCH EXAMPLE")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Predicates: {predicates}")
    print(f"k: {k}")
    print()

    # Perform LR-based adaptive search
    # This will automatically choose between PRE and POS based on the trained model
    query_embedding = search.embedder.encode_query(query)
    results = search.lr_based_adap_search(query_embedding, predicates, k)

    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)
    print(f"Found {len(results.results)} results")
    print(f"Is k satisfied: {results.is_k}")
    print()

    # Show results
    if results.results:
        results_df = results.to_df(show_cols=['item_id', 'title'])
        print(results_df.head(10))
    else:
        print("No results found.")

    # Show timing information
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN")
    print("=" * 80)
    timing_stats = timer.get_stats()
    if timing_stats:
        for section, time_ms in timing_stats.items():
            print(f"{section:20s}: {time_ms:8.2f} ms")

    search.close()


def compare_methods():
    """Compare different search methods on the same query"""

    query = "machine learning algorithms"
    predicates = [
        Predicate(key="token_count", value=400, operator=">="),
    ]
    k = 10

    methods = [
        ("LR-based Adaptive", "lr_based"),
        ("ADAP PRE", "adap_pre"),
        ("ADAP POS", "adap_pos"),
    ]

    print("=" * 80)
    print("COMPARING SEARCH METHODS")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Predicates: {predicates}")
    print(f"k: {k}")
    print()

    results_comparison = []

    for method_name, method_type in methods:
        timer = Timer()
        search = Search(timer=timer)
        query_embedding = search.embedder.encode_query(query)
        est_survivors = search.histo.estimate_survivors(predicates)

        if method_type == "lr_based":
            results = search.lr_based_adap_search(query_embedding, predicates, k)
        elif method_type == "adap_pre":
            results = search.adap_pre_search(query_embedding, predicates, k)
        elif method_type == "adap_pos":
            results = search.adap_pos_search(query_embedding, predicates, k, est_survivors)

        timing_stats = timer.get_stats()
        total_time = timing_stats.get('total', 0.0)

        results_comparison.append({
            'method': method_name,
            'num_results': len(results.results),
            'is_k': results.is_k,
            'total_time_ms': total_time,
        })

        search.close()

    print("\nRESULTS:")
    print("-" * 80)
    print(f"{'Method':<25} {'Results':<10} {'K OK':<10} {'Time (ms)':<15}")
    print("-" * 80)
    for res in results_comparison:
        print(f"{res['method']:<25} {res['num_results']:<10} {str(res['is_k']):<10} {res['total_time_ms']:>10.2f}")
    print("-" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic LR-based Adaptive Search")
    print("=" * 80 + "\n")
    example_lr_based_search()

    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Comparing Different Methods")
    print("=" * 80 + "\n")
    compare_methods()
