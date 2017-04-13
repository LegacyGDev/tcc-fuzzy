# unit tests for fuzzy.py

import unittest
import fuzzy

class DivideIntoFuzzyRegionsTestCase(unittest.TestCase):
    def test_is_output_a_list(self):
        self.assertIs(type(fuzzy.divide_into_fuzzy_regions([0,1,2,3,4,5,6,7,8,9],1)),type([]))

    def test_is_output_len(self):
        for i in range(5):
            self.assertEqual(len(fuzzy.divide_into_fuzzy_regions([0,1,2,3,4,5,6,7,8,9],i+1)),(2*(i+1))+1)


class DetermineDegreeAndAssignTestCase(unittest.TestCase):
    def test_is_output_a_tuple(self):
        self.assertIs(type(fuzzy.determine_degrees_and_assign(4,[(0,0,4.5),(0,4.5,9),(4.5,9,9)])),type(()))

    def test_is_only_region_output_a_tuple(self):
        self.assertIs(type(fuzzy.determine_degrees_and_assign(4,[(0,0,4.5),(0,4.5,9),(4.5,9,9)],only_regions=True)),type(()))


if __name__ == '__main__':
    unittest.main()
