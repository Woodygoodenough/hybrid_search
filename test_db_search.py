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
    
    def test_single_item_id_equals(self):
        """Test single item_id predicate: item_id = 1"""
        predicates = [Predicate(key="item_id", value=1, operator="=")]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 1)
        record = records.records[0]
        self.assertEqual(record.item_id, 1)
        self.assertEqual(record.title, "NFL controversies")
        self.assertEqual(record.token_count, 12479)
    
    def test_item_id_in_predicate(self):
        """Test item_id IN predicate with multiple values"""
        predicates = [Predicate(key="item_id", value=[1, 2, 3, 4, 5], operator="IN")]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 5)
        item_ids = [record.item_id for record in records.records]
        expected_ids = [1, 2, 3, 4, 5]
        self.assertEqual(sorted(item_ids), expected_ids)
        
        # Verify specific records
        record_dict = records.to_dict()
        self.assertEqual(record_dict[1].title, "NFL controversies")
        self.assertEqual(record_dict[2].title, "Devil Masami")
        self.assertEqual(record_dict[3].title, "Shinar")
    
    def test_item_id_range_predicate(self):
        """Test item_id range predicate: item_id <= 10"""
        predicates = [Predicate(key="item_id", value=10, operator="<=")]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 11)
        item_ids = [record.item_id for record in records.records]
        expected_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(sorted(item_ids), expected_ids)
        
        # Verify all records have item_id <= 10
        for record in records.records:
            self.assertLessEqual(record.item_id, 10)
    
    def test_item_id_greater_than_predicate(self):
        """Test item_id greater than predicate: item_id > 5"""
        predicates = [Predicate(key="item_id", value=5, operator=">")]
        records = self.db.predicates_search(predicates)
        
        # Should return all records with item_id > 5
        self.assertEqual(len(records), 149994)  # 150000 - 6 (0,1,2,3,4,5)
        
        # Verify all records have item_id > 5
        for record in records.records:
            self.assertGreater(record.item_id, 5)
    
    def test_item_id_with_other_predicates(self):
        """Test item_id IN combined with token_count predicate"""
        predicates = [
            Predicate(key="item_id", value=[1, 2, 3, 4, 5], operator="IN"),
            Predicate(key="token_count", value=1000, operator="<")
        ]
        records = self.db.predicates_search(predicates)
        
        self.assertEqual(len(records), 3)
        
        # Verify specific records and their token counts
        record_dict = records.to_dict()
        self.assertEqual(record_dict[2].token_count, 922)
        self.assertEqual(record_dict[3].token_count, 707)
        self.assertEqual(record_dict[4].token_count, 335)
        
        # Verify all records have token_count < 1000
        for record in records.records:
            self.assertLess(record.token_count, 1000)
    
    def test_item_id_nonexistent(self):
        """Test item_id predicate with non-existent ID"""
        predicates = [Predicate(key="item_id", value=999999, operator="=")]
        records = self.db.predicates_search(predicates)
        self.assertEqual(len(records), 0)
    
    def test_item_id_in_empty_list(self):
        """Test item_id IN predicate with empty list"""
        predicates = [Predicate(key="item_id", value=[], operator="IN")]
        records = self.db.predicates_search(predicates)
        self.assertEqual(len(records), 0)


if __name__ == "__main__":
    unittest.main()
