def test_imports():
    import app.config
    import app.llm  # noqa: F401
    assert app.config.settings is not None
