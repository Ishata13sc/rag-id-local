def test_imports():
    import api.main as app
    import src.ingest.ingest as ingest
    import src.index.build_index as build_index
    import src.models.embedder as embedder
    import src.storage.db as db
    assert app and ingest and build_index and embedder and db
