"""Read draft acceptance counters that Ollama omits from /api/generate."""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit


LLAMA_LOG = re.compile(r"draft acceptance(?: rate)?\s*=\s*[\d.]+\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\s*\)", re.I)
MLX_LOG = re.compile(r"speculative decode stats\b[^\r\n]*?\bdrafted=(\d+)\s+accepted=(\d+)", re.I)


def parse_log_counters(text: str) -> tuple[int, int] | None:
    """Return one unambiguous (drafted, accepted) pair from a request window."""
    matches: list[tuple[int, int]] = []
    for line in text.splitlines():
        llama = LLAMA_LOG.search(line)
        mlx = MLX_LOG.search(line)
        if llama:
            accepted, drafted = map(int, llama.groups())
        elif mlx:
            drafted, accepted = map(int, mlx.groups())
        else:
            continue
        if drafted > 0 and 0 <= accepted <= drafted:
            matches.append((drafted, accepted))
    return matches[0] if len(matches) == 1 else None


def _ssh(host: str, command: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
        raise ValueError("Invalid SSH host")
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command],
        capture_output=True, text=True, timeout=20, check=True,
    )
    return completed.stdout


class LogReader:
    """Local file, SSH file, systemd journal, or Docker container logs."""

    def __init__(self, source: str):
        self.source = source
        parsed = urlsplit(source)
        self.scheme = parsed.scheme.lower()
        if self.scheme in ("ssh", "journal", "docker"):
            if not parsed.hostname or not parsed.path.strip("/"):
                raise ValueError("Log source needs a host and a file, unit, or container")
            if parsed.port is not None:
                raise ValueError("Configure a custom SSH port in your SSH config")
            self.host = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
            self.target = parsed.path
        else:
            self.scheme = "file"
            self.path = Path(source).expanduser()

    def mark(self) -> int | float:
        if self.scheme == "file":
            return self.path.stat().st_size
        if self.scheme == "ssh":
            return int(_ssh(self.host, "stat -c %s -- " + shlex.quote(self.target)).strip())
        return time.time()

    def read_since(self, mark: int | float) -> str:
        if self.scheme == "file":
            with self.path.open("rb") as stream:
                stream.seek(int(mark) if self.path.stat().st_size >= mark else 0)
                return stream.read().decode("utf-8", errors="replace")
        if self.scheme == "ssh":
            size = int(_ssh(self.host, "stat -c %s -- " + shlex.quote(self.target)).strip())
            offset = int(mark) if size >= mark else 0
            return _ssh(self.host, "tail -c +" + str(offset + 1) + " -- " + shlex.quote(self.target))
        since = str(max(0, int(mark) - 1))
        target = shlex.quote(self.target.lstrip("/"))
        if self.scheme == "journal":
            command = "journalctl -u " + target + " --since " + shlex.quote("@" + since) + " --no-pager -o cat"
        else:
            command = "docker logs --since " + since + " " + target + " 2>&1"
        return _ssh(self.host, command)
