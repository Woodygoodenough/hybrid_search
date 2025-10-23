import unittest
from dbManagement import DbManagement, Predicate


class TestDbSearch(unittest.TestCase):
    def setUp(self):
        self.db = DbManagement()
    
    def test_single_numeric_predicate(self):
        """Test single numeric predicate: token_count < 150"""
        predicates = [Predicate(key="token_count", value=150, operator="<")]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 3406)
        for record in records:
            self.assertLess(record.token_count, 150)
    
    def test_single_date_predicate(self):
        """Test single date predicate: revdate >= 2025-01-01"""
        predicates = [Predicate(key="revdate", value="2025-01-01", operator=">=")]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 149947)
        for record in records:
            self.assertGreaterEqual(record.revdate, "2025-01-01")
    
    def test_multiple_predicates_and(self):
        """Test multiple predicates with AND logic"""
        predicates = [
            Predicate(key="token_count", value=1000, operator="<"),
            Predicate(key="token_count", value=100, operator=">")
        ]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 61486)
        for record in records:
            self.assertLess(record.token_count, 1000)
            self.assertGreater(record.token_count, 100)
    
    def test_empty_predicates(self):
        """Test empty predicates list returns all records"""
        records = self.db.predicates_search([])
        self.assertEqual(len(records), 150000)
    
    def test_no_matching_records(self):
        """Test predicate that matches no records"""
        predicates = [Predicate(key="token_count", value=1, operator="<")]
        records = self.db.predicates_search(predicates)
        self.assertEqual(len(records), 0)


if __name__ == "__main__":
    unittest.main()
