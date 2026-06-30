import unittest

from setuptools import find_packages


class TestPackaging(unittest.TestCase):
    def test_utils_package_is_discovered(self):
        packages = find_packages(
            exclude=[
                "tests",
                "tests_scripts",
                "benchmarks",
                "tutorials",
            ]
        )

        self.assertIn("swarmopt", packages)
        self.assertIn("swarmopt.utils", packages)


if __name__ == "__main__":
    unittest.main()
