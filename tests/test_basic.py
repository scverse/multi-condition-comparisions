import pytest

import multi_condition_comparisions


def test_package_has_version():
    assert multi_condition_comparisions.__version__ is not None


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.
