from pathlib import Path

from setuptools import find_packages


def test_find_packages_includes_algorithm_utilities():
    repo_root = Path(__file__).resolve().parents[2]
    packages = find_packages(
        where=str(repo_root),
        exclude=["tests", "tests_scripts", "benchmarks", "tutorials"]
    )

    assert "swarmopt" in packages
    assert "swarmopt.utils" in packages
