import os

# Set before api.py is imported. Values are fake; tests never call the real services.
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/rc_brain_test")
os.environ.setdefault("RETRIEVAL_API_KEY", "test-retrieval-key")
os.environ.setdefault("BASE44_API_KEY", "test-base44")
os.environ.setdefault("BASE44_APP_ID", "app-test")
os.environ["REINDEX_SCHEDULER"] = "false"
os.environ["REINDEX_ON_STARTUP"] = "false"
os.environ["REINDEX_INLINE"] = "true"
os.environ["RETRIEVAL_SCHEMA_ON_START"] = "false"
os.environ["ANN_INDEX_ON_START"] = "false"
os.environ["VOYAGE_MIN_INTERVAL_SECONDS"] = "0"
