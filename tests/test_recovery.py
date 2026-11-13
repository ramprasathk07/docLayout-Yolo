import io

import pytest

from indicdlp.data.prepare import read_manifest
from indicdlp.data.ranges import Budget, RangeReader
from tests.helpers import Response, Session


def test_torn_unicode_manifest_append_is_recovered(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b'{"id":"ok"}\n{"id":"\xe0\xa4')
    assert list(read_manifest(tmp_path)) == ["ok"]
    assert path.read_bytes() == b'{"id":"ok"}\n'


def test_corrupted_complete_manifest_line_is_not_ignored(tmp_path):
    (tmp_path / "manifest.jsonl").write_bytes(b"bad\n")
    with pytest.raises(ValueError):
        read_manifest(tmp_path)


def test_changed_remote_size_rejected_before_read():
    class BadSizeSession(Session):
        def get(self, *args, **kwargs):
            response = super().get(*args, **kwargs)
            response.headers["Content-Range"] = "bytes 0-1/999"
            return response

    session = BadSizeSession(b"abc")
    with pytest.raises(RuntimeError, match="Content-Range"):
        RangeReader("https://fixture", 3, session, Budget(10)).read(2)
    assert session.last_response.raw.tell() == 0


def test_truncated_range_counts_received_bytes():
    class TruncatedSession:
        def get(self, *args, **kwargs):
            response = Response(b"abcd", 0, 3)
            response.raw = io.BytesIO(b"ab")
            return response

    budget = Budget(4)
    with pytest.raises(OSError, match="Truncated"):
        RangeReader("https://fixture", 4, TruncatedSession(), budget).read(4)
    assert budget.used == 2
