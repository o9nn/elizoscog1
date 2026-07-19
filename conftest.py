"""
pytest configuration for the ElizaOS-OpenCog-GnuCash integration test suite.

Marks
-----
integration
    Tests that require external services (running AtomSpace server, ElizaOS
    server, or a real GnuCash file) to be present.  These are skipped in the
    default ``pytest`` run and must be opted-in explicitly::

        pytest -m integration

unit
    Pure unit tests with no external dependencies (default run).

Usage in test files
-------------------
    import pytest

    @pytest.mark.integration
    def test_needs_real_atomspace():
        ...

    def test_pure_logic():  # no mark needed — runs by default
        ...
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live external services "
        "(skipped unless -m integration is passed)"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as a pure unit test (always runs)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly requested."""
    if config.getoption("-m", default="") == "integration":
        return  # User explicitly asked for integration tests — run everything

    skip_integration = pytest.mark.skip(
        reason="Integration test skipped in default run — use: pytest -m integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
