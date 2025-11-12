#!/usr/bin/env python3
"""Comprehensive benchmark for k=[5, 10, 100, 1000, 10000] with proper histogram timing"""

from timer import _timer, reset_timings, save_run, clear_runs, time_section
from shared_dataclasses import Predicate
from search import Search
import pandas as pd

# Test queries with different selectivity
test_queries = [
    ('very_selective_0.1pct', [Predicate(key='item_id', value=list(range(0, 150)), operator='IN')]),
    ('selective_10pct', [Predicate(key='token_count', value=7000, operator='>')]),
    ('moderate_50pct', [Predicate(key='token_count', value=1000, operator='<')]),
    ('low_selectivity_90pct', [Predicate(key='token_count', value=4000, operator='<')]),
]

k_values = [5, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

print('='*80)
print('COMPREHENSIVE BENCHMARK: k=[5, 10, 100, 1000, 10000]')
print('='*80)

search = Search()
all_results = []

for query, preds in test_queries:
    print(f'\n{"="*80}')
    print(f'Query: {query}')
    print(f'Predicates: {preds}')
    print('='*80)
    
    embedding_query = search.embedder.encode_query(query)
  
    query_results = []
    
    for k in k_values:
        print(f'\n  Testing k={k}...', end=' ', flush=True)
        clear_runs()
        
        # Test 1: base_pre_search
        for _ in range(1):  # warmup
            _ = search.base_pre_search(embedding_query, preds, k)
        for _ in range(10):
            reset_timings()
            _ = search.base_pre_search(embedding_query, preds, k)
            save_run()
        
        # Test 2: adap_pre_search
        for _ in range(1):  # warmup
            _ = search.histo.estimate_survivors(preds)
            _ = search.adap_pre_search(embedding_query, preds, k)
        for _ in range(10):
            reset_timings()
            with time_section('histogram_estimation'):
                est_survivors = search.histo.estimate_survivors(preds)
            _ = search.adap_pre_search(embedding_query, preds, k)
            save_run()
        
        # Test 3: base_pos_search
        for _ in range(1):  # warmup
            _ = search.base_pos_search(embedding_query, preds, k)
        for _ in range(10):
            reset_timings()
            _ = search.base_pos_search(embedding_query, preds, k)
            save_run()
        
        # Test 4: adap_pos_search
        for _ in range(1):  # warmup
            est_survivors = search.histo.estimate_survivors(preds)
            _ = search.adap_pos_search(embedding_query, preds, k, est_survivors)
        for _ in range(10):
            reset_timings()
            with time_section('histogram_estimation'):
                est_survivors = search.histo.estimate_survivors(preds)
            _ = search.adap_pos_search(embedding_query, preds, k, est_survivors)
            save_run()
        
        # Process runs and create dataframe
        all_dfs = []
        method_runs_map = {
            "base_pre_search": (0, 10),
            "adap_pre_search": (10, 20),
            "base_pos_search": (20, 30),
            "adap_pos_search": (30, 40),
        }
        
        for method_name, (start_idx, end_idx) in method_runs_map.items():
            method_runs = _timer.runs[start_idx:end_idx]
            if method_runs:
                from timer import Timer
                temp_timer = Timer()
                temp_timer.runs = method_runs
                df = temp_timer.to_dataframe(use_averaged=True)
                if len(df) > 0:
                    df['k'] = k
                    all_dfs.append(df)
        
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            query_results.append((k, df))
        
        print('✓', end='', flush=True)
    
    all_results.append((query, preds, query_results))
    print()  # New line after all k values

search.close()

# Create comprehensive summary
print('\n' + '='*80)
print('COMPREHENSIVE RESULTS SUMMARY')
print('='*80)

# Create a summary dataframe
summary_data = []

for query, preds, query_results in all_results:
    for k, df in query_results:
        adap_pre_row = df[df['method'] == 'adap_pre_search']
        adap_pos_row = df[df['method'] == 'adap_pos_search']
        base_pre_row = df[df['method'] == 'base_pre_search']
        base_pos_row = df[df['method'] == 'base_pos_search']
        
        adap_pre_time = adap_pre_row['total_time'].iloc[0] if len(adap_pre_row) > 0 else 0
        adap_pos_time = adap_pos_row['total_time'].iloc[0] if len(adap_pos_row) > 0 else 0
        base_pre_time = base_pre_row['total_time'].iloc[0] if len(base_pre_row) > 0 else 0
        base_pos_time = base_pos_row['total_time'].iloc[0] if len(base_pos_row) > 0 else 0
        
        winner = "adap_pre" if adap_pre_time < adap_pos_time else "adap_pos"
        speedup = max(adap_pre_time, adap_pos_time) / min(adap_pre_time, adap_pos_time) if min(adap_pre_time, adap_pos_time) > 0 else 0
        
        summary_data.append({
            'query': query,
            'k': k,
            'adap_pre_ms': adap_pre_time,
            'adap_pos_ms': adap_pos_time,
            'base_pre_ms': base_pre_time,
            'base_pos_ms': base_pos_time,
            'winner': winner,
            'speedup': speedup,
        })

summary_df = pd.DataFrame(summary_data)

# Print detailed tables for each query
for query, preds, query_results in all_results:
    print(f'\n{"="*80}')
    print(f'Query: {query}')
    print('='*80)
    print(f'{'k':<8} {'adap_pre (ms)':<15} {'adap_pos (ms)':<15} {'base_pre (ms)':<15} {'base_pos (ms)':<15} {'Winner':<10} {'Speedup':<10}')
    print('-'*80)
    
    for k, df in query_results:
        adap_pre_row = df[df['method'] == 'adap_pre_search']
        adap_pos_row = df[df['method'] == 'adap_pos_search']
        base_pre_row = df[df['method'] == 'base_pre_search']
        base_pos_row = df[df['method'] == 'base_pos_search']
        
        adap_pre_time = adap_pre_row['total_time'].iloc[0] if len(adap_pre_row) > 0 else 0
        adap_pos_time = adap_pos_row['total_time'].iloc[0] if len(adap_pos_row) > 0 else 0
        base_pre_time = base_pre_row['total_time'].iloc[0] if len(base_pre_row) > 0 else 0
        base_pos_time = base_pos_row['total_time'].iloc[0] if len(base_pos_row) > 0 else 0
        
        winner = "adap_pre" if adap_pre_time < adap_pos_time else "adap_pos"
        speedup = max(adap_pre_time, adap_pos_time) / min(adap_pre_time, adap_pos_time) if min(adap_pre_time, adap_pos_time) > 0 else 0
        
        print(f'{k:<8} {adap_pre_time:<15.2f} {adap_pos_time:<15.2f} {base_pre_time:<15.2f} {base_pos_time:<15.2f} {winner:<10} {speedup:.2f}x')

# Print crossover analysis
print('\n' + '='*80)
print('CROSSOVER ANALYSIS: When does adap_pre outperform adap_pos?')
print('='*80)

for query, preds, query_results in all_results:
    print(f'\n{query}:')
    print('-'*80)
    
    crossover_found = False
    for k, df in query_results:
        adap_pre_row = df[df['method'] == 'adap_pre_search']
        adap_pos_row = df[df['method'] == 'adap_pos_search']
        
        adap_pre_time = adap_pre_row['total_time'].iloc[0] if len(adap_pre_row) > 0 else 0
        adap_pos_time = adap_pos_row['total_time'].iloc[0] if len(adap_pos_row) > 0 else 0
        
        if adap_pre_time > 0 and adap_pos_time > 0:
            winner = "adap_pre" if adap_pre_time < adap_pos_time else "adap_pos"
            speedup = max(adap_pre_time, adap_pos_time) / min(adap_pre_time, adap_pos_time)
            
            if winner == "adap_pre" and not crossover_found:
                print(f'  k={k}: adap_pre wins ({speedup:.2f}x faster) ← CROSSOVER POINT')
                crossover_found = True
            elif winner == "adap_pre":
                print(f'  k={k}: adap_pre wins ({speedup:.2f}x faster)')
            else:
                print(f'  k={k}: adap_pos wins ({speedup:.2f}x faster)')
    
    if not crossover_found:
        print('  adap_pos wins for all k values')

# Print scaling analysis
print('\n' + '='*80)
print('SCALING ANALYSIS: How does performance scale with k?')
print('='*80)

for query, preds, query_results in all_results:
    print(f'\n{query}:')
    print('-'*80)
    
    # Get times for adap_pre and adap_pos
    adap_pre_times = {}
    adap_pos_times = {}
    
    for k, df in query_results:
        adap_pre_row = df[df['method'] == 'adap_pre_search']
        adap_pos_row = df[df['method'] == 'adap_pos_search']
        
        adap_pre_times[k] = adap_pre_row['total_time'].iloc[0] if len(adap_pre_row) > 0 else 0
        adap_pos_times[k] = adap_pos_row['total_time'].iloc[0] if len(adap_pos_row) > 0 else 0
    
    # Calculate scaling factors
    print('  adap_pre_search scaling:')
    prev_time = None
    for k in k_values:
        if k in adap_pre_times and adap_pre_times[k] > 0:
            if prev_time:
                scale_factor = adap_pre_times[k] / prev_time
                print(f'    k={prev_k} → k={k}: {adap_pre_times[prev_k]:.2f}ms → {adap_pre_times[k]:.2f}ms ({scale_factor:.2f}x)')
            prev_time = adap_pre_times[k]
            prev_k = k
    
    print('  adap_pos_search scaling:')
    prev_time = None
    for k in k_values:
        if k in adap_pos_times and adap_pos_times[k] > 0:
            if prev_time:
                scale_factor = adap_pos_times[k] / prev_time
                print(f'    k={prev_k} → k={k}: {adap_pos_times[prev_k]:.2f}ms → {adap_pos_times[k]:.2f}ms ({scale_factor:.2f}x)')
            prev_time = adap_pos_times[k]
            prev_k = k

# Final summary table
print('\n' + '='*80)
print('FINAL SUMMARY TABLE')
print('='*80)
print(f'{'Query':<25} {'Selectivity':<12} {'k':<8} {'adap_pre':<12} {'adap_pos':<12} {'Winner':<10} {'Speedup':<10}')
print('-'*80)

for row in summary_df.itertuples():
    print(f'{row.query:<25} {row.selectivity:<12} {row.k:<8} {row.adap_pre_ms:<12.2f} {row.adap_pos_ms:<12.2f} {row.winner:<10} {row.speedup:.2f}x')

print('\n' + '='*80)
print('Benchmark complete!')
print('='*80)

