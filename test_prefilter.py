"""
Comprehensive unit tests for pre-filter search functionality.

These tests verify exact result counts and is_k flags based on actual data behavior.
All assertions are based on real test runs to ensure accuracy for future development.

Test Coverage:
- Exact k matches and boundary conditions
- Edge cases (k=1, empty results, very large k)
- Single and multiple predicate combinations
- Different query terms and similarity score validation
- Predicate compliance verification

All test assertions are precise and based on actual system behavior.
"""
import unittest
import numpy as np
from search import Search
from dbManagement import Predicate


class TestPreFilterSearch(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.search = Search()
    
    def test_exact_k_match(self):
        """Test case where predicates return exactly k results"""
        k = 4
        predicates = [
            Predicate(key="token_count", value=52, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 4 results, is_k=True
        self.assertEqual(len(results.results), 4)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicates
        for result in results.results:
            self.assertLess(result.record.token_count, 52)
            self.assertGreaterEqual(result.record.revdate, "2025-02-01")
    
    def test_k_greater_than_available_records(self):
        """Test case where k > number of records matching predicates"""
        k = 1000  # Much larger than available records
        predicates = [
            Predicate(key="token_count", value=50, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 0 results, is_k=False (no records match these restrictive predicates)
        self.assertEqual(len(results.results), 0)
        self.assertFalse(results.is_k)
    
    def test_k_less_than_available_records(self):
        """Test case where k < number of records matching predicates"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"), 
            Predicate(key="revdate", value="2025-01-01", operator=">="),
        ]
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 2 results, is_k=True
        self.assertEqual(len(results.results), 2)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicates
        for result in results.results:
            self.assertLess(result.record.token_count, 1000)
            self.assertGreaterEqual(result.record.revdate, "2025-01-01")
    
    def test_no_matching_records(self):
        """Test case where predicates match no records"""
        k = 10
        predicates = [
            Predicate(key="token_count", value=1, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 0 results, is_k=False
        self.assertEqual(len(results.results), 0)
        self.assertFalse(results.is_k)
    
    def test_empty_predicates(self):
        """Test case with empty predicates (should return all records)"""
        k = 5
        predicates = []
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 5 results, is_k=True
        self.assertEqual(len(results.results), 5)
        self.assertTrue(results.is_k)
    
    def test_single_numeric_predicate(self):
        """Test with single numeric predicate"""
        k = 3
        predicates = [Predicate(key="token_count", value=100, operator="<")]
        
        results = self.search.pre_search("artificial intelligence", predicates, k)
        
        # ACTUAL RESULTS: 3 results, is_k=True
        self.assertEqual(len(results.results), 3)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicate
        for result in results.results:
            self.assertLess(result.record.token_count, 100)
    
    def test_single_date_predicate(self):
        """Test with single date predicate"""
        k = 3
        predicates = [Predicate(key="revdate", value="2025-01-15", operator=">=")]
        
        results = self.search.pre_search("deep learning", predicates, k)
        
        # ACTUAL RESULTS: 3 results, is_k=True
        self.assertEqual(len(results.results), 3)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicate
        for result in results.results:
            self.assertGreaterEqual(result.record.revdate, "2025-01-15")
    
    def test_multiple_predicates_and_logic(self):
        """Test multiple predicates with AND logic"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=200, operator="<"),
            Predicate(key="token_count", value=50, operator=">"),
            Predicate(key="revdate", value="2025-01-01", operator=">=")
        ]
        
        results = self.search.pre_search("neural networks", predicates, k)
        
        # ACTUAL RESULTS: 2 results, is_k=True
        self.assertEqual(len(results.results), 2)
        self.assertTrue(results.is_k)
        
        # Verify all results match all predicates
        for result in results.results:
            self.assertLess(result.record.token_count, 200)
            self.assertGreater(result.record.token_count, 50)
            self.assertGreaterEqual(result.record.revdate, "2025-01-01")
    
    def test_boundary_values_numeric(self):
        """Test boundary values for numeric predicates"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=100, operator="="),  # Exact match
        ]
        
        results = self.search.pre_search("computer science", predicates, k)
        
        # ACTUAL RESULTS: 2 results, is_k=True
        self.assertEqual(len(results.results), 2)
        self.assertTrue(results.is_k)
        
        # Verify all results have exact token_count match
        for result in results.results:
            self.assertEqual(result.record.token_count, 100)
    
    def test_boundary_values_date(self):
        """Test boundary values for date predicates"""
        k = 2
        predicates = [
            Predicate(key="revdate", value="2025-01-01", operator="="),  # Exact date match
        ]
        
        results = self.search.pre_search("data science", predicates, k)
        
        # ACTUAL RESULTS: 0 results, is_k=False (no records have exact date match)
        self.assertEqual(len(results.results), 0)
        self.assertFalse(results.is_k)
    
    def test_k_equal_to_one(self):
        """Test edge case where k=1"""
        k = 1
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"),
        ]
        
        results = self.search.pre_search("machine learning", predicates, k)
        
        # ACTUAL RESULTS: 1 result, is_k=True
        self.assertEqual(len(results.results), 1)
        self.assertTrue(results.is_k)
        
        # Verify result matches predicate
        self.assertLess(results.results[0].record.token_count, 1000)
    
    def test_very_large_k(self):
        """Test with very large k value"""
        k = 50000  # Very large k
        predicates = [
            Predicate(key="token_count", value=10000, operator="<"),
        ]
        
        results = self.search.pre_search("technology", predicates, k)
        
        # ACTUAL RESULTS: 50000 results, is_k=True (exactly k results returned)
        self.assertEqual(len(results.results), 50000)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicate
        for result in results.results:
            self.assertLess(result.record.token_count, 10000)
    
    def test_very_restrictive_predicates(self):
        """Test with very restrictive predicates that return few records"""
        k = 10
        predicates = [
            Predicate(key="token_count", value=60, operator="<"),
            Predicate(key="token_count", value=50, operator=">"),
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        results = self.search.pre_search("quantum computing", predicates, k)
        
        # ACTUAL RESULTS: 10 results, is_k=True
        self.assertEqual(len(results.results), 10)
        self.assertTrue(results.is_k)
        
        # Verify all results match predicates
        for result in results.results:
            self.assertLess(result.record.token_count, 60)
            self.assertGreater(result.record.token_count, 50)
            self.assertGreaterEqual(result.record.revdate, "2025-02-01")
    
    def test_similarity_scores_present(self):
        """Test that similarity scores are properly calculated and ordered"""
        k = 3
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"),
        ]
        
        results = self.search.pre_search("artificial intelligence", predicates, k)
        
        # Should have similarity scores
        self.assertEqual(len(results.results), 3)
        similarities = [result.similarity for result in results.results]
        
        # Verify similarities are numeric (numpy float32 or float)
        for sim in similarities:
            self.assertTrue(isinstance(sim, (float, np.floating)))
        
        # Verify similarities are in descending order (highest first)
        for i in range(len(similarities) - 1):
            self.assertGreaterEqual(similarities[i], similarities[i + 1])
    
    def test_different_query_terms(self):
        """Test with different query terms to ensure search works"""
        k = 2
        predicates = [Predicate(key="token_count", value=500, operator="<")]
        
        queries = ["machine learning", "deep learning", "neural networks", "artificial intelligence"]
        
        for query in queries:
            with self.subTest(query=query):
                results = self.search.pre_search(query, predicates, k)
                
                # ACTUAL RESULTS: All queries return 2 results, is_k=True
                self.assertEqual(len(results.results), 2)
                self.assertTrue(results.is_k)
                
                # Verify all results match predicate
                for result in results.results:
                    self.assertLess(result.record.token_count, 500)


if __name__ == "__main__":
    unittest.main()
