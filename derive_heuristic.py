#!/usr/bin/env python3
"""Derive heuristic for choosing adap_pre vs adap_pos based on k and est_survivors"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Extract crossover points from benchmark results
# Format: (est_survivors, selectivity_ratio, crossover_k)
crossover_data = [
    (150, 0.001, float('inf')),  # Always adap_pre
    (15000, 0.10, 1000),  # Crossover at k=1000
    (75000, 0.50, 2000),  # Crossover at k=2000
    (135000, 0.90, 5000),  # Crossover at k=5000
]

print('='*80)
print('HEURISTIC DERIVATION: adap_pre vs adap_pos')
print('='*80)
print()

# Analyze patterns
print('Crossover Points:')
print('-'*80)
print(f'{'Est Survivors':<15} {'Selectivity':<12} {'Crossover k':<15} {'k/est_survivors':<15} {'k*selectivity':<15}')
print('-'*80)

for est, sel, k_cross in crossover_data:
    if k_cross == float('inf'):
        k_ratio = float('inf')
        k_sel = float('inf')
    else:
        k_ratio = k_cross / est
        k_sel = k_cross * sel
    print(f'{est:<15} {sel:<12.3f} {str(k_cross):<15} {str(k_ratio):<15} {str(k_sel):<15}')

print()
print('='*80)
print('PATTERN ANALYSIS')
print('='*80)

# Try different formulas
print('\n1. k / est_survivors ratio:')
for est, sel, k_cross in crossover_data:
    if k_cross != float('inf'):
        ratio = k_cross / est
        print(f'   est={est}, k_cross={k_cross}, ratio={ratio:.4f}')

print('\n2. k * selectivity:')
for est, sel, k_cross in crossover_data:
    if k_cross != float('inf'):
        product = k_cross * sel
        print(f'   est={est}, k_cross={k_cross}, product={product:.1f}')

print('\n3. k / (est_survivors / N):')
N = 150000
for est, sel, k_cross in crossover_data:
    if k_cross != float('inf'):
        ratio = k_cross / (est / N)
        print(f'   est={est}, k_cross={k_cross}, ratio={ratio:.1f}')

# Fit a curve to find the relationship
print('\n' + '='*80)
print('CURVE FITTING')
print('='*80)

# Use data points where crossover exists
x_data = [est for est, sel, k in crossover_data if k != float('inf')]
y_data = [k for est, sel, k in crossover_data if k != float('inf')]

print(f'\nData points: est_survivors = {x_data}, crossover_k = {y_data}')

# Try linear fit: k = a * est_survivors + b
def linear_func(x, a, b):
    return a * x + b

try:
    popt_linear, _ = curve_fit(linear_func, x_data, y_data)
    print(f'\nLinear fit: k = {popt_linear[0]:.6f} * est_survivors + {popt_linear[1]:.2f}')
    print('   Predicted crossovers:')
    for est in x_data:
        pred_k = linear_func(est, *popt_linear)
        actual_k = next(k for e, s, k in crossover_data if e == est)
        print(f'     est={est}: predicted={pred_k:.0f}, actual={actual_k}, error={abs(pred_k-actual_k):.0f}')
except:
    print('Linear fit failed')

# Try power fit: k = a * est_survivors^b
def power_func(x, a, b):
    return a * np.power(x, b)

try:
    popt_power, _ = curve_fit(power_func, x_data, y_data)
    print(f'\nPower fit: k = {popt_power[0]:.6f} * est_survivors^{popt_power[1]:.4f}')
    print('   Predicted crossovers:')
    for est in x_data:
        pred_k = power_func(est, *popt_power)
        actual_k = next(k for e, s, k in crossover_data if e == est)
        print(f'     est={est}: predicted={pred_k:.0f}, actual={actual_k}, error={abs(pred_k-actual_k):.0f}')
except:
    print('Power fit failed')

# Try ratio-based: k = a * est_survivors / (1 - est_survivors/N)
print('\n' + '='*80)
print('RATIO-BASED HEURISTIC')
print('='*80)

# Look at k / est_survivors ratio
ratios = [k / est for est, sel, k in crossover_data if k != float('inf')]
print(f'\nk / est_survivors ratios: {ratios}')
avg_ratio = np.mean(ratios)
print(f'Average ratio: {avg_ratio:.4f}')

# Look at k * selectivity
products = [k * sel for est, sel, k in crossover_data if k != float('inf')]
print(f'\nk * selectivity products: {products}')
avg_product = np.mean(products)
print(f'Average product: {avg_product:.1f}')

# Derive heuristic
print('\n' + '='*80)
print('PROPOSED HEURISTIC')
print('='*80)

# Based on analysis, let's use a simple threshold
# For very selective (< 1%), always use adap_pre
# For others, use a threshold based on k and est_survivors

print('\nHeuristic 1: Simple threshold based on k/est_survivors')
threshold_ratio = np.mean(ratios)
print(f'  Use adap_pre if: k / est_survivors >= {threshold_ratio:.4f}')
print(f'  Use adap_pos if: k / est_survivors < {threshold_ratio:.4f}')
print(f'  Special case: est_survivors < {N * 0.01} (1%) → always use adap_pre')

print('\nHeuristic 2: Threshold based on k * selectivity')
threshold_product = np.mean(products)
print(f'  Use adap_pre if: k * (est_survivors / N) >= {threshold_product:.1f}')
print(f'  Use adap_pos if: k * (est_survivors / N) < {threshold_product:.1f}')
print(f'  Special case: est_survivors < {N * 0.01} (1%) → always use adap_pre')

# More sophisticated: piecewise linear
print('\nHeuristic 3: Piecewise linear (more accurate)')
# Fit linear to log scale
log_x = np.log(x_data)
log_y = np.log(y_data)
popt_log, _ = curve_fit(linear_func, log_x, log_y)
print(f'  Log-linear fit: log(k) = {popt_log[0]:.4f} * log(est) + {popt_log[1]:.4f}')
print(f'  Formula: k = {np.exp(popt_log[1]):.2f} * est_survivors^{popt_log[0]:.4f}')

# Test the heuristics
print('\n' + '='*80)
print('HEURISTIC VALIDATION')
print('='*80)

def heuristic1(k, est_survivors, N=150000):
    """Simple ratio-based heuristic"""
    if est_survivors < N * 0.01:  # < 1%
        return 'adap_pre'
    return 'adap_pre' if k / est_survivors >= threshold_ratio else 'adap_pos'

def heuristic2(k, est_survivors, N=150000):
    """Product-based heuristic"""
    if est_survivors < N * 0.01:  # < 1%
        return 'adap_pre'
    selectivity = est_survivors / N
    return 'adap_pre' if k * selectivity >= threshold_product else 'adap_pos'

def heuristic3(k, est_survivors, N=150000):
    """Piecewise linear heuristic"""
    if est_survivors < N * 0.01:  # < 1%
        return 'adap_pre'
    threshold_k = np.exp(popt_log[1]) * np.power(est_survivors, popt_log[0])
    return 'adap_pre' if k >= threshold_k else 'adap_pos'

print('\nTesting heuristics on known crossover points:')
print('-'*80)
print(f'{'Est':<10} {'k':<10} {'Actual':<12} {'Heuristic1':<12} {'Heuristic2':<12} {'Heuristic3':<12}')
print('-'*80)

test_cases = [
    (150, 100, 'adap_pre'),
    (15000, 500, 'adap_pos'),
    (15000, 1500, 'adap_pre'),
    (75000, 1000, 'adap_pos'),
    (75000, 2500, 'adap_pre'),
    (135000, 4000, 'adap_pos'),
    (135000, 6000, 'adap_pre'),
]

for est, k, actual in test_cases:
    h1 = heuristic1(k, est)
    h2 = heuristic2(k, est)
    h3 = heuristic3(k, est)
    print(f'{est:<10} {k:<10} {actual:<12} {h1:<12} {h2:<12} {h3:<12}')

# Final recommendation
print('\n' + '='*80)
print('FINAL RECOMMENDED HEURISTIC')
print('='*80)
print()
print('def choose_method(k, est_survivors, N=150000):')
print('    """Choose between adap_pre_search and adap_pos_search"""')
print('    # Very selective queries: always use adap_pre')
print(f'    if est_survivors < N * 0.01:  # < 1%')
print('        return "adap_pre_search"')
print('    ')
print('    # For other queries, use threshold based on k and est_survivors')
print(f'    threshold_k = {np.exp(popt_log[1]):.2f} * est_survivors ** {popt_log[0]:.4f}')
print('    if k >= threshold_k:')
print('        return "adap_pre_search"')
print('    else:')
print('        return "adap_pos_search"')
print()

# Alternative simpler heuristic
print('='*80)
print('SIMPLER ALTERNATIVE HEURISTIC')
print('='*80)
print()
print('def choose_method_simple(k, est_survivors, N=150000):')
print('    """Simpler heuristic using ratio threshold"""')
print('    # Very selective queries: always use adap_pre')
print(f'    if est_survivors < N * 0.01:  # < 1%')
print('        return "adap_pre_search"')
print('    ')
print(f'    # Use adap_pre if k is large relative to est_survivors')
print(f'    if k / est_survivors >= {threshold_ratio:.4f}:')
print('        return "adap_pre_search"')
print('    else:')
print('        return "adap_pos_search"')

