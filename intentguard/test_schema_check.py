import unittest
from schema_check import check_call

class TestSchemaCheck(unittest.TestCase):
    def test_real_endpoint_valid(self):
        res = check_call("GET", "/api/orders/")
        self.assertTrue(res["valid"])
        
    def test_real_endpoint_missing_slash(self):
        res = check_call("GET", "/api/orders")
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "Missing trailing slash normalised.")
        
    def test_invalid_route(self):
        res = check_call("POST", "/api/orders/999/refund/")
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "No such route.")
        
    def test_literal_placeholder(self):
        res = check_call("GET", "/api/orders/{id}/")
        self.assertFalse(res["valid"])
        
    def test_invalid_method(self):
        # /api/orders/999/cancel/ is POST-only
        res = check_call("GET", "/api/orders/999/cancel/")
        self.assertFalse(res["valid"])
        self.assertIn("not allowed", res["reason"])

if __name__ == "__main__":
    unittest.main()
