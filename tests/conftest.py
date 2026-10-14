import os
import pytest
import tempfile

@pytest.fixture(autouse=True)
def set_audit_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="audit_")
    os.close(fd)
    os.environ["INTENTGUARD_AUDIT_PATH"] = path
    yield
    os.unlink(path)
