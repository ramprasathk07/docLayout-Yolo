"""Seekable HTTP reads with a hard payload budget and no whole-file fallback."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limit: int
    used: int = 0

    def check(self, requested: int) -> None:
        if requested < 0 or self.used + requested > self.limit:
            raise BudgetExceeded(
                f"Transfer budget exceeded: {self.used:,} bytes used, "
                f"{requested:,} requested, {self.limit:,} allowed. "
                "Preparation is resumable; increase --max-transfer-mib on the server."
            )


def http_session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept-Encoding"] = "identity"
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class RangeReader(io.RawIOBase):
    """Only issues bounded byte ranges; refuses HTTP 200 before consuming a body.

    Requests strips Authorization on cross-host redirects. HF's signed CDN URLs
    consequently work without forwarding the HF token to the CDN.
    """

    def __init__(self, url, size, session, budget):
        super().__init__()
        self.url, self.size, self.session, self.budget = url, size, session, budget
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset + (
            0 if whence == 0 else self.position if whence == 1 else self.size
        )
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read on closed stream")
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count <= 0:
            return b""
        self.budget.check(count)
        start, end = self.position, self.position + count - 1
        with self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(15, 120),
        ) as response:
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "HF access denied. Accept the dataset gate and set HF_TOKEN."
                )
            if response.status_code != 206:
                raise RuntimeError(
                    f"Server returned HTTP {response.status_code}; refusing a non-range "
                    "response to prevent a whole-shard download."
                )
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or tuple(map(int, match.groups())) != (start, end, self.size):
                raise RuntimeError(
                    "Unexpected Content-Range; remote file may have changed"
                )
            result = bytearray()
            # Bounded raw reads also protect against a server sending excess bytes.
            while len(result) < count:
                chunk = response.raw.read(min(1024 * 1024, count - len(result)))
                if not chunk:
                    raise OSError("Truncated HTTP range; rerun preparation to resume")
                self.budget.used += len(chunk)
                result.extend(chunk)
        self.position += count
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
