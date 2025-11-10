import pytest
import time
import json
from statistics import mean
from search import Search
from shared_dataclasses import Predicate

TIMING_DATA = {}


@pytest.fixture(scope="module")
def search_engine():
    """Initialize Search engine once per module."""
    with Search() as search:
        yield search


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function call and return results + duration."""
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        success = True
        error_msg = None
    except Exception as e:
        result = None
        success = False
        error_msg = str(e)
    end = time.perf_counter()
    duration = end - start
    return result, duration, success, error_msg


def unwrap_results(results):
    """Handle wrapper types like HSearchResults."""
    if hasattr(results, "results"):
        return results.results
    return results


def record_timing(strategy, duration):
    """Store timing results for later comparison."""
    TIMING_DATA.setdefault(strategy, []).append(duration)


def log_test_result(test_name, query, predicates, method, results, duration, success, error_msg):
    """Structured debug logging for each test."""
    unwrapped = unwrap_results(results)
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print(f"Query: {query}")
    print(f"Method: {method or 'default'}")
    print("Predicates:")
    print(json.dumps([p.__dict__ for p in predicates], indent=2))
    print(f"Duration: {duration:.6f}s")
    print(f"Success: {success}")

    if not success:
        print(f"Error: {error_msg}")
    else:
        if isinstance(unwrapped, list):
            preview = unwrapped[:2]
            print(f"Results Preview ({len(unwrapped)} total):")
            print(json.dumps(preview, indent=2, default=str))
        else:
            print(f"Unexpected result type: {type(unwrapped)}")
            print(str(unwrapped)[:200])
    print("=" * 80 + "\n")



def test_default_search_verbose(search_engine):
    """Default search test with timing."""
    query = "machine learning algorithms"
    predicates = []
    results, duration, success, error_msg = measure_time(search_engine.search, query, predicates, 5)

    log_test_result("Default Search", query, predicates, None, results, duration, success, error_msg)
    record_timing("default", duration)

    unwrapped = unwrap_results(results)
    assert success, f"Search failed: {error_msg}"
    assert isinstance(unwrapped, list), f"Unexpected result type: {type(unwrapped)}"
    assert len(unwrapped) <= 5


def test_filtered_search_verbose(search_engine):
    """Search with metadata filters and debug info."""
    query = "AI research"
    predicates = [
        Predicate(key="token_count", value=500, operator="<"),
        Predicate(key="revdate", value="2025-01-01", operator=">="),
    ]
    results, duration, success, error_msg = measure_time(search_engine.search, query, predicates, 5)

    log_test_result("Filtered Search", query, predicates, None, results, duration, success, error_msg)
    record_timing("filtered", duration)

    unwrapped = unwrap_results(results)
    assert success, f"Search failed: {error_msg}"
    assert isinstance(unwrapped, list)
    assert len(unwrapped) <= 5


@pytest.mark.parametrize("method", ["pre_search", "post_search"])
def test_strategy_search_verbose(search_engine, method):
    """Compare pre_search and post_search strategies."""
    query = "deep learning"
    predicates = [
        Predicate(key="token_count", value=500, operator="<"),
        Predicate(key="revdate", value="2025-01-01", operator=">="),
    ]
    results, duration, success, error_msg = measure_time(
        search_engine.search, query, predicates, 5, method=method
    )

    log_test_result(f"Strategy: {method}", query, predicates, method, results, duration, success, error_msg)
    record_timing(method, duration)

    unwrapped = unwrap_results(results)
    assert success, f"{method} search failed: {error_msg}"
    assert isinstance(unwrapped, list)
    assert len(unwrapped) <= 5


# -----------------------------
# Timing summary (prints once)
# -----------------------------

def teardown_module(module):
    """Print summary of timing comparisons after all tests run."""
    print("\n" + "=" * 80)
    print("SEARCH STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)

    if not TIMING_DATA:
        print("No timing data collected.")
        return

    header = f"{'Strategy':<15} {'Runs':<5} {'Avg Time (s)':<15} {'Min (s)':<15} {'Max (s)':<15}"
    print(header)
    print("-" * len(header))

    for strategy, durations in TIMING_DATA.items():
        avg_time = mean(durations)
        print(f"{strategy:<15} {len(durations):<5} {avg_time:<15.6f} {min(durations):<15.6f} {max(durations):<15.6f}")

    print("=" * 80 + "\n")
