import unittest
import sys
import re
import isca_tools.thesis.test as thesis

def suite_thesis() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(thesis.TestLapseIntegral))
    return suite

def suite_all():
    suite = suite_thesis()
    # suite.addTest(suite_setup())
    return suite

if __name__ == '__main__':
    suite = suite_all()
    unittest.main(defaultTest='suite', exit=True)