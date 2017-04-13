# unit tests for fuzzy.py

import unittest
import fuzzy

class DivideIntoFuzzyRegionsTestCase(unittest.TestCase):
    def test_is_output_a_list(self):
        self.assertIs(type(fuzzy.divide_into_fuzzy_regions([0,1,2,3,4,5,6,7,8,9],1)),type([]))
    def test_is_output_len(self):
        for i in range(5):
            self.assertEqual(len(fuzzy.divide_into_fuzzy_regions([0,1,2,3,4,5,6,7,8,9],i+1)),(2*(i+1))+1)

if __name__ == '__main__':
    unittest.main()
