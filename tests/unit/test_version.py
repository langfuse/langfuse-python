from importlib.metadata import version

import langfuse


def test_package_version_matches_distribution_metadata():
    assert langfuse.__version__ == version("langfuse")
