import varistar


def test_version():
    """Check that the version is accessible."""
    assert varistar.__version__ is not None


def test_placeholder():
    """A simple placeholder test that always passes."""
    assert True
