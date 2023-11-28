import pytest

import multi_condition_comparisions


def test_package_has_version():
    assert multi_condition_comparisions.__version__ is not None

@pytest.param("method")
def test_de(method):
    raise NotImplementedError
