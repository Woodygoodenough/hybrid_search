"""
Unit tests for both pre-filter and post-filter search functionality.

These tests verify that both strategies return the same number of results and is_k flags.
For cases where is_k is false (not enough results found), exact result matching is verified.
For normal cases, only result count matching is checked.

Test Coverage:
- Exact k matches and boundary conditions
- Edge cases (k=1, empty results, very large k)
- Single and multiple predicate combinations
- Different query terms
- Performance comparison between strategies

Both strategies should return identical results when is_k is false, and same counts otherwise.
"""
import unittest
from search import Search
from dbManagement import Predicate


class TestHybridSearchStrategies(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.search = Search()
    
    def _test_both_strategies(self, query: str, predicates: list, k: int, expected_count: int, expected_is_k: bool):
        """Helper method to test both pre-filter and post-filter strategies."""
        # Test pre-filter strategy
        pre_results = self.search.pre_search(query, predicates, k)
        
        # Test post-filter strategy
        post_results = self.search.post_search(query, predicates, k)

        # Test hybrid strategy
        hybrid_results = self.search.hybrid_search(query, predicates, k)
        
        # Both strategies should return the same number of results
        self.assertEqual(len(pre_results.results), expected_count)
        self.assertEqual(len(post_results.results), expected_count)
        self.assertEqual(len(hybrid_results.results), expected_count)
        # Both strategies should have the same is_k flag
        self.assertEqual(pre_results.is_k, expected_is_k)
        self.assertEqual(post_results.is_k, expected_is_k)
        self.assertEqual(hybrid_results.is_k, expected_is_k)
        # Only check for exact matches when is_k is false (not enough results found)
        # In this case, both strategies should return all available survivors
        if not expected_is_k and expected_count > 0:
            pre_item_ids = [result.record.item_id for result in pre_results.results]
            post_item_ids = [result.record.item_id for result in post_results.results]
            self.assertEqual(pre_item_ids, post_item_ids, "Pre-filter and post-filter should return identical item_ids when is_k is false")
        
        return pre_results, post_results, hybrid_results
    
    def test_exact_k_match(self):
        """Test case where predicates return exactly k results"""
        k = 4
        predicates = [
            Predicate(key="token_count", value=52, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 4, True)
    
    def test_k_greater_than_available_records(self):
        """Test case where k > number of records matching predicates"""
        k = 1000  # Much larger than available records
        predicates = [
            Predicate(key="token_count", value=50, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 0, False)
    
    def test_k_less_than_available_records(self):
        """Test case where k < number of records matching predicates"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"), 
            Predicate(key="revdate", value="2025-01-01", operator=">="),
        ]
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 2, True)
    
    def test_no_matching_records(self):
        """Test case where predicates match no records"""
        k = 10
        predicates = [
            Predicate(key="token_count", value=1, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 0, False)
    
    def test_empty_predicates(self):
        """Test case with empty predicates (should return all records)"""
        k = 5
        predicates = []
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 5, True)
    
    def test_single_numeric_predicate(self):
        """Test with single numeric predicate"""
        k = 3
        predicates = [Predicate(key="token_count", value=100, operator="<")]
        
        # Test both strategies
        self._test_both_strategies("artificial intelligence", predicates, k, 3, True)
    
    def test_single_date_predicate(self):
        """Test with single date predicate"""
        k = 3
        predicates = [Predicate(key="revdate", value="2025-01-15", operator=">=")]
        
        # Test both strategies
        self._test_both_strategies("deep learning", predicates, k, 3, True)
    
    def test_multiple_predicates_and_logic(self):
        """Test multiple predicates with AND logic"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=200, operator="<"),
            Predicate(key="token_count", value=50, operator=">"),
            Predicate(key="revdate", value="2025-01-01", operator=">=")
        ]
        
        # Test both strategies
        self._test_both_strategies("neural networks", predicates, k, 2, True)
    
    def test_boundary_values_numeric(self):
        """Test boundary values for numeric predicates"""
        k = 2
        predicates = [
            Predicate(key="token_count", value=100, operator="="),  # Exact match
        ]
        
        # Test both strategies
        self._test_both_strategies("computer science", predicates, k, 2, True)
    
    def test_boundary_values_date(self):
        """Test boundary values for date predicates"""
        k = 2
        predicates = [
            Predicate(key="revdate", value="2025-01-01", operator="="),  # Exact date match
        ]
        
        # Test both strategies
        self._test_both_strategies("data science", predicates, k, 0, False)
    
    def test_k_equal_to_one(self):
        """Test edge case where k=1"""
        k = 1
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"),
        ]
        
        # Test both strategies
        self._test_both_strategies("machine learning", predicates, k, 1, True)
    
    def test_very_large_k(self):
        """Test with very large k value"""
        k = 50000  # Very large k
        predicates = [
            Predicate(key="token_count", value=10000, operator="<"),
        ]
        
        # Test both strategies
        self._test_both_strategies("technology", predicates, k, 50000, True)
    
    def test_very_restrictive_predicates(self):
        """Test with very restrictive predicates that return few records"""
        k = 10
        predicates = [
            Predicate(key="token_count", value=60, operator="<"),
            Predicate(key="token_count", value=50, operator=">"),
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        # Test both strategies
        self._test_both_strategies("quantum computing", predicates, k, 10, True)

    
    def test_different_query_terms(self):
        """Test with different query terms to ensure search works"""
        k = 2
        predicates = [Predicate(key="token_count", value=500, operator="<")]
        
        queries = ["machine learning", "deep learning", "neural networks", "artificial intelligence"]
        
        for query in queries:
            with self.subTest(query=query):
                # Test both strategies
                self._test_both_strategies(query, predicates, k, 2, True)


if __name__ == "__main__":
    unittest.main()
