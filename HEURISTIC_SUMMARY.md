# Heuristic for Choosing adap_pre_search vs adap_pos_search

## Overview

Based on comprehensive benchmarks with k=[5, 10, 100, 1000, 2000, ..., 10000] and selectivity levels [0.1%, 10%, 50%, 90%], we derived a heuristic to automatically choose between `adap_pre_search` and `adap_pos_search`.

## Key Findings

### Crossover Points
- **Very selective (0.1%)**: Always use `adap_pre_search` (no crossover)
- **Selective (10%)**: Crossover at k=1000
- **Moderate (50%)**: Crossover at k=2000
- **Low selectivity (90%)**: Crossover at k=5000

### Pattern
The crossover point increases with both `k` and `est_survivors`, following a power-law relationship:
```
threshold_k = 1.49 * est_survivors^0.6686
```

## Implementation

### Recommended: Accurate Method (94.1% accuracy)

```python
def choose_method(k, est_survivors, N=150000):
    """Choose between adap_pre_search and adap_pos_search"""
    # Very selective queries: always use adap_pre
    if est_survivors < N * 0.01:  # < 1%
        return "adap_pre_search"
    
    # For other queries, use threshold based on k and est_survivors
    threshold_k = 1.49 * (est_survivors ** 0.6686)
    if k >= threshold_k:
        return "adap_pre_search"
    else:
        return "adap_pos_search"
```

### Alternative: Simple Method (82.4% accuracy)

```python
def choose_method_simple(k, est_survivors, N=150000):
    """Simpler heuristic using ratio threshold"""
    # Very selective queries: always use adap_pre
    if est_survivors < N * 0.01:  # < 1%
        return "adap_pre_search"
    
    # Use adap_pre if k is large relative to est_survivors
    if k / est_survivors >= 0.0435:
        return "adap_pre_search"
    else:
        return "adap_pos_search"
```

## Usage

```python
from heuristic_method_selector import choose_method

# Example 1: Small k, moderate selectivity → use adap_pos
method = choose_method(k=10, est_survivors=15000)  # Returns "adap_pos_search"

# Example 2: Large k, moderate selectivity → use adap_pre
method = choose_method(k=2000, est_survivors=15000)  # Returns "adap_pre_search"

# Example 3: Very selective → always use adap_pre
method = choose_method(k=100, est_survivors=150)  # Returns "adap_pre_search"
```

## Validation Results

- **Accurate method**: 16/17 correct (94.1% accuracy)
- **Simple method**: 14/17 correct (82.4% accuracy)

Errors occur at crossover boundaries, which is expected due to the discrete nature of the decision boundary.

## Rationale

1. **Very selective queries (< 1%)**: Database filtering is fast on small sets, so `adap_pre_search` always wins.

2. **Moderate to low selectivity**: 
   - Small k: `adap_pos_search` wins (avoids expensive upfront DB filtering)
   - Large k: `adap_pre_search` wins (DB filtering cost amortized, FAISS search with whitelist scales better)

3. **Crossover point**: Shifts right (toward higher k) as selectivity increases, following a power-law relationship.

## Files

- `heuristic_method_selector.py`: Implementation with both methods
- `benchmark_all_k.py`: Comprehensive benchmark script
- `benchmark_k_detailed.txt`: Full benchmark results

