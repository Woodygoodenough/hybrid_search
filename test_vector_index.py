"""
Unit tests for FaissIVFIndex.search method.

Tests cover:
- Normal search cases with expected results
- Edge cases: k > survivors (requesting more results than available)
- Edge cases: no results found
- ID filtering functionality
- Result count verification using len()
- AnnSearchResults object methods and properties
- Proper filtering of invalid results (no -1 values)

All tests use the existing trained index and verify that:
1. len(results) returns the correct number of valid results
2. No -1 values are present in the results (filtered out automatically)
3. Results are properly ordered by similarity (descending)
4. ID filtering works correctly for both existing and non-existing IDs
"""
import unittest
import warnings
import numpy as np
from vector_index import FaissIVFIndex, Embedder


class TestFaissIVFIndexSearch(unittest.TestCase):
    """Test cases for FaissIVFIndex.search method."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Suppress FAISS warnings during testing
        warnings.filterwarnings("ignore", category=UserWarning)

        # Load the existing trained index instead of creating a new one
        self.index = FaissIVFIndex.load("index.faiss.new")

        # Create embedder for queries
        self.embedder = Embedder()

        # Create test queries
        self.query1 = self.embedder.encode_query("machine learning")
        self.query2 = self.embedder.encode_query("artificial intelligence")
        self.query3 = self.embedder.encode_query("random unrelated text for testing")

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_normal_case_successful_search(self):
        """Test normal case: search finds expected number of results."""
        k = 5
        results = self.index.search(self.query1, k)

        # Verify we got the expected number of results
        self.assertEqual(len(results), k)
        self.assertEqual(len(results.item_ids), k)
        self.assertEqual(len(results.distances), k)

        # Verify all results are valid (no -1 values)
        self.assertTrue(np.all(results.item_ids != -1))
        self.assertTrue(np.all(results.distances >= -1.0))  # Distances should be >= -1 (cosine similarity)
        self.assertTrue(np.all(results.distances <= 1.0))   # Distances should be <= 1 (cosine similarity)

        # Verify distances are in descending order (most similar first)
        self.assertTrue(np.all(results.distances[:-1] >= results.distances[1:]))

    def test_k_equals_one(self):
        """Test edge case: k = 1."""
        k = 1
        results = self.index.search(self.query2, k)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results.item_ids), 1)
        self.assertEqual(len(results.distances), 1)

        # Should find the closest match
        self.assertNotEqual(results.item_ids[0], -1)
        self.assertGreaterEqual(results.distances[0], -1.0)
        self.assertLessEqual(results.distances[0], 1.0)

    def test_k_greater_than_available_matches(self):
        """Test edge case: k > survivors (request more results than actually found)."""
        # Search with filtering for non-existing IDs - should return empty results
        non_existing_ids = [99999, 99998, 99997]
        k = 10
        results = self.index.search(self.query1, k, item_ids=non_existing_ids)

        # Should return empty results when no matches found
        self.assertEqual(len(results), 0)
        self.assertEqual(len(results.item_ids), 0)
        self.assertEqual(len(results.distances), 0)

    def test_no_results_found(self):
        """Test edge case: no results found (random query with strict filtering)."""
        k = 5

        # Search with filtering for specific IDs that may not exist
        specific_ids = [999999, 999998, 999997]  # Very unlikely to exist
        results = self.index.search(self.query3, k, item_ids=specific_ids)

        # Results should be empty since no matches found
        self.assertEqual(len(results), 0)
        self.assertEqual(len(results.item_ids), 0)
        self.assertEqual(len(results.distances), 0)

    def test_id_filtering_normal_case(self):
        """Test ID filtering: search within specific item IDs."""
        # First get some results without filtering to find existing IDs
        unfiltered_results = self.index.search(self.query1, 10)

        if len(unfiltered_results) >= 3:
            # Select a subset of existing IDs
            target_ids = unfiltered_results.item_ids[:3].tolist()
            k = 10  # Request more than available to test filtering

            results = self.index.search(self.query1, k, item_ids=target_ids)

            # Results should only contain items from target_ids
            if len(results) > 0:
                self.assertTrue(np.all(np.isin(results.item_ids, target_ids)))
                self.assertEqual(len(results.item_ids), len(results))
                self.assertEqual(len(results.distances), len(results))

                # Verify no -1 values (already handled by AnnSearchResults)
                self.assertTrue(np.all(results.item_ids != -1))
        else:
            # If no results, skip this test
            self.skipTest("No results found for unfiltered search")

    def test_id_filtering_partial_matches(self):
        """Test ID filtering: when only some requested IDs exist."""
        # First get some results without filtering to find existing IDs
        unfiltered_results = self.index.search(self.query1, 10)

        if len(unfiltered_results) >= 2:
            # Mix of existing and non-existing IDs
            existing_ids = unfiltered_results.item_ids[:2].tolist()
            mixed_ids = existing_ids + [999999, 999998]  # Add non-existing IDs
            k = 5

            results = self.index.search(self.query1, k, item_ids=mixed_ids)

            # Should only return results for existing IDs
            if len(results) > 0:
                self.assertTrue(np.all(np.isin(results.item_ids, existing_ids)))
                self.assertLessEqual(len(results), 2)  # At most 2 results (the existing IDs)
                self.assertEqual(len(results.item_ids), len(results))
        else:
            # If no results, skip this test
            self.skipTest("No results found for unfiltered search")

    def test_results_object_methods(self):
        """Test AnnSearchResults object methods and properties."""
        k = 3
        results = self.index.search(self.query1, k)

        # Test len() method
        self.assertEqual(len(results), k)

        # Test to_valid_dict() method
        valid_dict = results.to_dict()
        self.assertIsInstance(valid_dict, dict)
        self.assertEqual(len(valid_dict), len(results))

        # All keys should be valid item IDs (no -1)
        self.assertTrue(all(item_id != -1 for item_id in valid_dict.keys()))

        # All values should be distances
        self.assertTrue(all(-1.0 <= dist <= 1.0 for dist in valid_dict.values()))

    def test_nprobe_parameter(self):
        """Test nprobe parameter functionality."""
        k = 5

        # Test with default nprobe
        results_default = self.index.search(self.query1, k)

        # Test with custom nprobe
        results_custom = self.index.search(self.query1, k, nprobe=1)

        # Both should return valid results
        self.assertEqual(len(results_default), k)
        self.assertEqual(len(results_custom), k)

        # Results may differ due to different nprobe values, but should be valid
        self.assertTrue(np.all(results_default.item_ids != -1))
        self.assertTrue(np.all(results_custom.item_ids != -1))

    def test_filtering_removes_invalid_results(self):
        """Test that invalid results (-1 values) are properly filtered out."""
        # Test with a large k to potentially trigger padding
        k = 1000  # Much larger than available results
        results = self.index.search(self.query1, k)

        # Results should be filtered to only valid items
        self.assertEqual(len(results), len(results.item_ids))
        self.assertEqual(len(results), len(results.distances))

        # All item_ids should be valid (no -1)
        self.assertTrue(np.all(results.item_ids != -1))

        # All distances should be in valid range
        self.assertTrue(np.all(results.distances >= -1.0))
        self.assertTrue(np.all(results.distances <= 1.0))

        # Results should be ordered by similarity (descending)
        if len(results) > 1:
            self.assertTrue(np.all(results.distances[:-1] >= results.distances[1:]))


if __name__ == '__main__':
    unittest.main()
