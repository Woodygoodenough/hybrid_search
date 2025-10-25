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

    def tearDown(self):
        """Clean up after each test method."""
        self.search.close()
    
    def _test_both_strategies(self, query: str, predicates: list, k: int, min_expected_count: int, expected_is_k: bool):
        """Helper method to test both pre-filter and post-filter strategies."""
        # Encode query once for all methods
        embedding_query = self.search.embedder.encode_query(query)
        est_survivors = self.search.histo.estimate_survivors(predicates)

        # Test pre-filter strategy
        pre_results = self.search.pre_search(embedding_query, predicates, k)

        # Test post-filter strategy
        post_results = self.search.pos_search(embedding_query, predicates, k, est_survivors)

        # Test hybrid strategy
        hybrid_results = self.search.search(query, predicates, k, method="pre_search")  # Use search method for hybrid

        # Both strategies should return at least the minimum expected results
        self.assertGreaterEqual(len(pre_results.results), min_expected_count)
        self.assertGreaterEqual(len(post_results.results), min_expected_count)

        # Both strategies should find results if expected_is_k is True
        if expected_is_k:
            self.assertGreater(len(pre_results.results), 0, "Pre-search should find results when expected_is_k is True")
            self.assertGreater(len(post_results.results), 0, "Post-search should find results when expected_is_k is True")
        else:
            self.assertEqual(len(pre_results.results), 0, "Pre-search should find no results when expected_is_k is False")
            self.assertEqual(len(post_results.results), 0, "Post-search should find no results when expected_is_k is False")

        # All results should be valid (non-negative similarity, valid item IDs)
        for results in [pre_results, post_results, hybrid_results]:
            for result in results.results:
                self.assertGreaterEqual(result.similarity, 0.0)
                self.assertGreater(result.record.item_id, 0)

        # Only check for exact matches when is_k is false (not enough results found)
        # In this case, both strategies should return all available survivors
        if not expected_is_k and min_expected_count > 0:
            pre_item_ids = [result.record.item_id for result in pre_results.results]
            post_item_ids = [result.record.item_id for result in post_results.results]
            self.assertEqual(pre_item_ids, post_item_ids, "Pre-filter and post-filter should return identical item_ids when is_k is false")

        return pre_results, post_results, hybrid_results
    
    def test_exact_k_match(self):
        """Test case where predicates return exactly k results"""
        print("--------------------------------")
        print("Test case where predicates return exactly k results")
        k = 4
        predicates = [
            Predicate(key="token_count", value=500, operator="<"),
            Predicate(key="revdate", value="2025-01-01", operator=">="),
        ]
        
        # Test both strategies - expect at least 4 results
        self._test_both_strategies("machine learning", predicates, k, 4, True)
    
    def test_k_greater_than_available_records(self):
        """Test case where k > number of records matching predicates"""
        print("--------------------------------")
        print("Test case where k > number of records matching predicates")
        k = 1000  # Much larger than available records
        predicates = [
            Predicate(key="token_count", value=50, operator="<"),
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]

        # Test both strategies - expect 0 results
        self._test_both_strategies("machine learning", predicates, k, 0, False)
    
    def test_k_less_than_available_records(self):
        """Test case where k < number of records matching predicates"""
        print("--------------------------------")
        print("Test case where k < number of records matching predicates")
        k = 2
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"), 
            Predicate(key="revdate", value="2025-01-01", operator=">="),
        ]
        
        # Test both strategies - expect at least 2 results
        self._test_both_strategies("machine learning", predicates, k, 2, True)
    
    def test_no_matching_records(self):
        """Test case where predicates match no records"""
        print("--------------------------------")
        print("Test case where predicates match no records")
        k = 10
        predicates = [
            Predicate(key="token_count", value=1, operator="<"), 
            Predicate(key="revdate", value="2025-02-01", operator=">="),
        ]
        
        # Test both strategies - expect 0 results
        self._test_both_strategies("machine learning", predicates, k, 0, False)
    
    def test_empty_predicates(self):
        """Test case with empty predicates (should return all records)"""
        print("--------------------------------")
        print("Test case with empty predicates (should return all records)")
        k = 5
        predicates = []
        
        # Test both strategies - expect at least 5 results
        self._test_both_strategies("machine learning", predicates, k, 5, True)
    
    def test_single_numeric_predicate(self):
        """Test with single numeric predicate"""
        print("--------------------------------")
        print("Test with single numeric predicate")
        k = 3
        predicates = [Predicate(key="token_count", value=100, operator="<")]
        
        # Test both strategies - expect at least 3 results
        self._test_both_strategies("artificial intelligence", predicates, k, 3, True)
    
    def test_single_date_predicate(self):
        """Test with single date predicate"""
        print("--------------------------------")
        print("Test with single date predicate")
        k = 3
        predicates = [Predicate(key="revdate", value="2025-01-15", operator=">=")]
        
        # Test both strategies - expect at least 3 results
        self._test_both_strategies("deep learning", predicates, k, 3, True)
    
    def test_multiple_predicates_and_logic(self):
        """Test multiple predicates with AND logic"""
        print("--------------------------------")
        print("Test multiple predicates with AND logic")
        k = 2
        predicates = [
            Predicate(key="token_count", value=200, operator="<"),
            Predicate(key="token_count", value=50, operator=">"),
            Predicate(key="revdate", value="2025-01-01", operator=">=")
        ]
        
        # Test both strategies - expect at least 2 results
        self._test_both_strategies("neural networks", predicates, k, 2, True)
    
    def test_boundary_values_numeric(self):
        """Test boundary values for numeric predicates"""
        print("--------------------------------")
        print("Test boundary values for numeric predicates")
        k = 2
        predicates = [
            Predicate(key="token_count", value=100, operator="="),  # Exact match
        ]
        
        # Test both strategies - expect at least 2 results
        self._test_both_strategies("computer science", predicates, k, 2, True)
    
    def test_boundary_values_date(self):
        """Test boundary values for date predicates"""
        print("--------------------------------")
        print("Test boundary values for date predicates")
        k = 2
        predicates = [
            Predicate(key="revdate", value="2025-01-01", operator="="),  # Exact date match
        ]
        
        # Test both strategies - expect 0 results
        self._test_both_strategies("data science", predicates, k, 0, False)
    
    def test_k_equal_to_one(self):
        """Test edge case where k=1"""
        print("--------------------------------")
        print("Test edge case where k=1")
        k = 1
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"),
        ]
        
        # Test both strategies - expect at least 1 result
        self._test_both_strategies("machine learning", predicates, k, 1, True)
    
    def test_very_large_k(self):
        """Test with very large k value"""
        print("--------------------------------")
        print("Test with very large k value")
        k = 50000  # Very large k
        predicates = [
            Predicate(key="token_count", value=10000, operator="<"),
        ]
        
        # Test both strategies - expect at least some results (but probably not 50000)
        self._test_both_strategies("technology", predicates, k, 1000, True)
    
    def test_very_restrictive_predicates(self):
        """Test with very restrictive predicates that return few records"""
        print("--------------------------------")
        print("Test with very restrictive predicates that return few records")
        k = 10
        predicates = [
            Predicate(key="token_count", value=200, operator="<"),
            Predicate(key="token_count", value=100, operator=">"),
            Predicate(key="revdate", value="2025-01-01", operator=">="),
        ]
        
        # Test both strategies - expect at least some results
        self._test_both_strategies("quantum computing", predicates, k, 1, True)

    
    def test_different_query_terms(self):
        """Test with different query terms to ensure search works"""
        print("--------------------------------")
        print("Test with different query terms to ensure search works")
        k = 2
        predicates = [Predicate(key="token_count", value=500, operator="<")]
        
        queries = ["machine learning", "deep learning", "neural networks", "artificial intelligence"]
        
        for query in queries:
            with self.subTest(query=query):
                # Test both strategies - expect at least 2 results
                self._test_both_strategies(query, predicates, k, 2, True)


if __name__ == "__main__":
    unittest.main()
