import importlib


def test_app_imports():
    app = importlib.import_module("src.app")
    assert hasattr(app, "main")
