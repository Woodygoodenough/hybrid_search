#!/usr/bin/env python3
"""
Heuristic for choosing between adap_pre_search and adap_pos_search
based on k and est_survivors.

Derived from comprehensive benchmarks with k=[5,10,100,1000,2000,...,10000]
and selectivity levels [0.1%, 10%, 50%, 90%].
"""

from typing import Literal


def choose_method(
    k: int,
    est_survivors: int,
    N: int = 150000,
    method: Literal["simple", "accurate"] = "accurate"
) -> Literal["adap_pre_search", "adap_pos_search"]:
    """
    Choose between adap_pre_search and adap_pos_search based on k and est_survivors.
    
    Args:
        k: Number of results to return
        est_survivors: Estimated number of records matching predicates
        N: Total number of records in database (default: 150000)
        method: "simple" for ratio-based heuristic, "accurate" for power-law heuristic
    
    Returns:
        "adap_pre_search" or "adap_pos_search"
    
    Heuristic Logic:
        1. Very selective queries (< 1% survivors): Always use adap_pre_search
        2. For other queries:
           - Simple: Use adap_pre if k/est_survivors >= 0.0435
           - Accurate: Use adap_pre if k >= 1.49 * est_survivors^0.6686
    
    Examples:
        >>> choose_method(k=10, est_survivors=15000)  # 10% selectivity
        'adap_pos_search'
        >>> choose_method(k=1000, est_survivors=15000)  # 10% selectivity
        'adap_pre_search'
        >>> choose_method(k=100, est_survivors=150)  # 0.1% selectivity
        'adap_pre_search'
    """
    # Very selective queries: always use adap_pre
    if est_survivors < N * 0.01:  # < 1%
        return "adap_pre_search"
    
    if method == "simple":
        # Simple ratio-based heuristic
        # Threshold derived from average k/est_survivors ratio at crossover points
        threshold_ratio = 0.0435
        if k / est_survivors >= threshold_ratio:
            return "adap_pre_search"
        else:
            return "adap_pos_search"
    
    else:  # method == "accurate"
        # Power-law heuristic: k = 1.49 * est_survivors^0.6686
        # Derived from log-linear fit of crossover points
        threshold_k = 1.49 * (est_survivors ** 0.6686)
        if k >= threshold_k:
            return "adap_pre_search"
        else:
            return "adap_pos_search"


def choose_method_simple(k: int, est_survivors: int, N: int = 150000) -> Literal["adap_pre_search", "adap_pos_search"]:
    """
    Simplified heuristic using ratio threshold.
    
    This is a convenience wrapper for choose_method(k, est_survivors, N, method="simple").
    """
    return choose_method(k, est_survivors, N, method="simple")


# Validation against known crossover points
if __name__ == "__main__":
    print("="*80)
    print("HEURISTIC VALIDATION")
    print("="*80)
    
    # Known crossover points from benchmarks
    test_cases = [
        # (k, est_survivors, expected, description)
        (10, 150, "adap_pre", "Very selective (0.1%), small k"),
        (100, 150, "adap_pre", "Very selective (0.1%), medium k"),
        (1000, 150, "adap_pre", "Very selective (0.1%), large k"),
        
        (10, 15000, "adap_pos", "Selective (10%), small k"),
        (100, 15000, "adap_pos", "Selective (10%), medium k"),
        (1000, 15000, "adap_pre", "Selective (10%), crossover point"),
        (2000, 15000, "adap_pre", "Selective (10%), large k"),
        
        (10, 75000, "adap_pos", "Moderate (50%), small k"),
        (100, 75000, "adap_pos", "Moderate (50%), medium k"),
        (1000, 75000, "adap_pos", "Moderate (50%), medium-large k"),
        (2000, 75000, "adap_pre", "Moderate (50%), crossover point"),
        (3000, 75000, "adap_pre", "Moderate (50%), large k"),
        
        (10, 135000, "adap_pos", "Low selectivity (90%), small k"),
        (100, 135000, "adap_pos", "Low selectivity (90%), medium k"),
        (1000, 135000, "adap_pos", "Low selectivity (90%), medium-large k"),
        (5000, 135000, "adap_pre", "Low selectivity (90%), crossover point"),
        (10000, 135000, "adap_pre", "Low selectivity (90%), very large k"),
    ]
    
    print("\nTesting 'accurate' method:")
    print("-"*80)
    print(f"{'k':<8} {'est_surv':<10} {'Expected':<12} {'Predicted':<12} {'Match':<8} {'Description'}")
    print("-"*80)
    
    correct = 0
    total = 0
    
    for k, est, expected, desc in test_cases:
        predicted = choose_method(k, est, method="accurate")
        # Normalize expected to match return format
        expected_normalized = f"{expected}_search"
        match = "✓" if predicted == expected_normalized else "✗"
        if predicted == expected_normalized:
            correct += 1
        total += 1
        print(f"{k:<8} {est:<10} {expected:<12} {predicted:<12} {match:<8} {desc}")
    
    print("-"*80)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    print("\nTesting 'simple' method:")
    print("-"*80)
    print(f"{'k':<8} {'est_surv':<10} {'Expected':<12} {'Predicted':<12} {'Match':<8} {'Description'}")
    print("-"*80)
    
    correct_simple = 0
    total_simple = 0
    
    for k, est, expected, desc in test_cases:
        predicted = choose_method(k, est, method="simple")
        # Normalize expected to match return format
        expected_normalized = f"{expected}_search"
        match = "✓" if predicted == expected_normalized else "✗"
        if predicted == expected_normalized:
            correct_simple += 1
        total_simple += 1
        print(f"{k:<8} {est:<10} {expected:<12} {predicted:<12} {match:<8} {desc}")
    
    print("-"*80)
    print(f"Accuracy: {correct_simple}/{total_simple} ({100*correct_simple/total_simple:.1f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("Use 'accurate' method for better precision, especially for moderate selectivity.")
    print("Use 'simple' method for easier implementation and understanding.")
    print("="*80)

