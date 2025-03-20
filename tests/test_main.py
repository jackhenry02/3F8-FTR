import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Simply checking if it runs without errors
        self.assertIsNone(main())

if __name__ == '__main__':
    unittest.main()
