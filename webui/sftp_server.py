#!/usr/bin/env python3
"""Read-only SFTP service for ICEBERG atlas exports."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any
from urllib.parse import quote

import yaml
from werkzeug.security import check_password_hash

try:
    import asyncssh
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    asyncssh = None  # type: ignore[assignment]

try:
    from .vfs import AtlasVirtualFS, VfsEntry, VfsNotDirectoryError, VfsNotFoundError, VfsPermissionError, role_can_access_nist
except ImportError:  # pragma: no cover - direct script execution from webui/
    from vfs import AtlasVirtualFS, VfsEntry, VfsNotDirectoryError, VfsNotFoundError, VfsPermissionError, role_can_access_nist


def _admin_emails_from_env() -> set[str]:
    return {
        email.strip().lower()
        for email in os.environ.get("ICEBERG_ADMIN_EMAILS", "").split(",")
        if email.strip()
    }


_SFTP_ANALYTICS_INIT_DONE: set[Path] = set()


def _analytics_db_path(users_file: Path | None) -> Path | None:
    if "ICEBERG_ANALYTICS_DB" in os.environ:
        return Path(os.environ["ICEBERG_ANALYTICS_DB"])
    if users_file is not None:
        return users_file.parent / "analytics.db"
    return None


def _analytics_conn(db_path: Path | None) -> sqlite3.Connection | None:
    if db_path is None:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.DatabaseError:
        pass
    return conn


def _ensure_sftp_analytics_schema(db_path: Path | None) -> None:
    if db_path is None or db_path in _SFTP_ANALYTICS_INIT_DONE:
        return
    conn = _analytics_conn(db_path)
    if conn is None:
        return
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    ip      TEXT,
                    path    TEXT,
                    method  TEXT,
                    user_id TEXT,
                    source  TEXT DEFAULT 'web'
                )
                """
            )
            request_cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
            if "source" not in request_cols:
                conn.execute("ALTER TABLE requests ADD COLUMN source TEXT DEFAULT 'web'")
            conn.execute("UPDATE requests SET source = 'web' WHERE source IS NULL OR source = ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts  ON requests(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ip  ON requests(ip)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_source ON requests(source)")
        _SFTP_ANALYTICS_INIT_DONE.add(db_path)
    except Exception:
        pass
    finally:
        conn.close()


def _sftp_log_path(action: str, path: str | bytes | None = None) -> str:
    if isinstance(path, bytes):
        path = path.decode("utf-8", errors="replace")
    path = str(path or "/").replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    return f"/sftp/{action}{quote(path, safe='/._-')}"[:2048]


def log_sftp_request(
    users_file: Path | None,
    username: str,
    ip: str,
    action: str,
    path: str | bytes | None = None,
) -> None:
    db_path = _analytics_db_path(users_file)
    try:
        _ensure_sftp_analytics_schema(db_path)
        conn = _analytics_conn(db_path)
        if conn is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                "INSERT INTO requests (ts, ip, path, method, user_id, source) VALUES (?,?,?,?,?,?)",
                (ts, ip, _sftp_log_path(action, path), "SFTP", username, "sftp"),
            )
    except Exception:
        pass
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass


def _peer_ip_from_extra_info(obj: Any) -> str:
    candidates = [obj]
    try:
        candidates.append(obj.get_connection())
    except Exception:
        pass
    conn = getattr(obj, "_conn", None)
    if conn is not None:
        candidates.append(conn)

    for candidate in candidates:
        try:
            peer = candidate.get_extra_info("peername")
        except Exception:
            peer = None
        if isinstance(peer, tuple) and peer:
            return str(peer[0])
        if isinstance(peer, str):
            return peer
    return ""


def _load_users(users_file: Path | None) -> dict[str, Any]:
    if users_file is None or not users_file.exists():
        return {}
    try:
        with users_file.open("r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _save_users(users_file: Path | None, users: dict[str, Any]) -> None:
    if users_file is None:
        return
    tmp_fd, tmp_path = tempfile.mkstemp(dir=users_file.parent, suffix=".yaml.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            yaml.safe_dump(users, fh, default_flow_style=False)
        os.replace(tmp_path, users_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_record(users: dict[str, Any], email: str) -> dict[str, Any] | None:
    raw = users.get(email)
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"password": raw, "role": "user", "created": ""}
    if isinstance(raw, dict):
        return raw
    return None


def check_sftp_credentials(users_file: Path | None, email: str, password: str) -> bool:
    users = _load_users(users_file)
    record = _get_record(users, email.strip().lower())
    if record is None:
        return False
    try:
        return check_password_hash(record["password"], password)
    except Exception:
        return False


def record_sftp_login(users_file: Path | None, email: str) -> None:
    normalized = email.strip().lower()
    users = _load_users(users_file)
    raw = users.get(normalized)
    record = _get_record(users, normalized)
    if record is None:
        return
    record["last_login"] = datetime.now(timezone.utc).isoformat()
    if isinstance(raw, str):
        record.setdefault("role", "user")
        record.setdefault("created", "")
    users[normalized] = record
    _save_users(users_file, users)


def role_for_email(users_file: Path | None, email: str, admin_emails: set[str]) -> str:
    normalized = email.strip().lower()
    if normalized in admin_emails:
        return "admin"
    users = _load_users(users_file)
    record = _get_record(users, normalized)
    if record is None:
        return "user"
    return str(record.get("role", "user"))


class IcebergSSHServer(asyncssh.SSHServer if asyncssh else object):  # type: ignore[misc,valid-type]
    def __init__(self, users_file: Path | None):
        self.users_file = users_file
        self.client_ip = ""

    def connection_made(self, conn) -> None:
        self.client_ip = _peer_ip_from_extra_info(conn)

    def begin_auth(self, username: str) -> bool:
        return True

    def password_auth_supported(self) -> bool:
        return True

    def public_key_auth_supported(self) -> bool:
        return False

    def validate_password(self, username: str, password: str) -> bool:
        normalized = username.strip().lower()
        ok = check_sftp_credentials(self.users_file, normalized, password)
        if ok:
            log_sftp_request(self.users_file, normalized, self.client_ip, "auth", "/")
            try:
                record_sftp_login(self.users_file, normalized)
            except Exception:
                pass
        return ok

    def connection_requested(self, *args, **kwargs):
        return False

    def server_requested(self, *args, **kwargs):
        return False

    def unix_connection_requested(self, *args, **kwargs):
        return False

    def unix_server_requested(self, *args, **kwargs):
        return False


class IcebergSFTPServer(asyncssh.SFTPServer if asyncssh else object):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        chan,
        vfs: AtlasVirtualFS,
        users_file: Path | None,
        admin_emails: set[str],
    ):
        super().__init__(chan)
        self.vfs = vfs
        self.users_file = users_file
        self.admin_emails = admin_emails
        self.username = (chan.get_extra_info("username") or "").strip().lower()
        self.client_ip = _peer_ip_from_extra_info(chan)
        self.role = role_for_email(users_file, self.username, admin_emails)

    @property
    def include_nist(self) -> bool:
        return role_can_access_nist(self.role, self.username, self.admin_emails)

    @staticmethod
    def _to_attrs(entry: VfsEntry):
        mtime = int(entry.mtime or 0)
        return asyncssh.SFTPAttrs(
            size=entry.size,
            permissions=entry.mode,
            atime=mtime,
            mtime=mtime,
        )

    @classmethod
    def _to_name(cls, entry: VfsEntry):
        return asyncssh.SFTPName(entry.name.encode("utf-8"), attrs=cls._to_attrs(entry))

    @staticmethod
    def _raise_sftp_error(exc: Exception) -> None:
        if isinstance(exc, VfsPermissionError):
            raise asyncssh.SFTPPermissionDenied(str(exc)) from exc
        if isinstance(exc, VfsNotFoundError):
            raise asyncssh.SFTPNoSuchFile(str(exc)) from exc
        if isinstance(exc, VfsNotDirectoryError):
            raise asyncssh.SFTPFailure(str(exc)) from exc
        raise asyncssh.SFTPFailure(str(exc)) from exc

    def realpath(self, path):
        try:
            entry = self.vfs.stat(path or "/", self.include_nist)
            log_sftp_request(self.users_file, self.username, self.client_ip, "realpath", entry.path)
            return str(entry.path).encode("utf-8")
        except Exception as exc:
            self._raise_sftp_error(exc)

    def stat(self, path):
        try:
            entry = self.vfs.stat(path, self.include_nist)
            log_sftp_request(self.users_file, self.username, self.client_ip, "stat", entry.path)
            return self._to_attrs(entry)
        except Exception as exc:
            self._raise_sftp_error(exc)

    def lstat(self, path):
        return self.stat(path)

    def listdir(self, path):
        try:
            entries = self.vfs.listdir(path or "/", self.include_nist)
            log_sftp_request(self.users_file, self.username, self.client_ip, "listdir", path or "/")
            return [self._to_name(entry) for entry in entries]
        except Exception as exc:
            self._raise_sftp_error(exc)

    def open(self, path, pflags, attrs):
        write_flags = 0
        for name, fallback in (
            ("FXF_WRITE", 0x00000002),
            ("FXF_APPEND", 0x00000004),
            ("FXF_CREAT", 0x00000008),
            ("FXF_TRUNC", 0x00000010),
            ("FXF_EXCL", 0x00000020),
        ):
            write_flags |= getattr(asyncssh, name, fallback)
        if pflags & write_flags:
            raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")
        try:
            file_obj = self.vfs.open_read(path, self.include_nist)
            log_sftp_request(self.users_file, self.username, self.client_ip, "open", path)
            return file_obj
        except Exception as exc:
            self._raise_sftp_error(exc)

    def setstat(self, path, attrs):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def fsetstat(self, file_obj, attrs):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def remove(self, path):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def mkdir(self, path, attrs):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def rmdir(self, path):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def rename(self, oldpath, newpath):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def posix_rename(self, oldpath, newpath):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def symlink(self, oldpath, newpath):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")

    def link(self, oldpath, newpath):
        raise asyncssh.SFTPPermissionDenied("The ICEBERG atlas SFTP service is read-only")


async def _serve(args: argparse.Namespace) -> None:
    if asyncssh is None:
        raise RuntimeError("asyncssh is required. Install/update the webui environment first.")

    users_file = Path(args.users_file) if args.users_file else None
    admin_emails = _admin_emails_from_env()
    vfs = AtlasVirtualFS(args.atlas_dir, args.nist_atlas_dir)
    host_keys = [args.host_key] if args.host_key else None

    await asyncssh.create_server(
        lambda: IcebergSSHServer(users_file),
        args.host,
        args.port,
        server_host_keys=host_keys,
        sftp_factory=lambda chan: IcebergSFTPServer(chan, vfs, users_file, admin_emails),
    )
    await asyncio.Future()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ICEBERG read-only SFTP service.")
    parser.add_argument("--host", default=os.environ.get("ICEBERG_SFTP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ICEBERG_SFTP_PORT", "2222")))
    parser.add_argument("--host-key", default=os.environ.get("ICEBERG_SFTP_HOST_KEY", ""))
    parser.add_argument("--users-file", default=os.environ.get("ICEBERG_USERS_FILE", ""))
    parser.add_argument("--atlas-dir", default=os.environ.get("MSPRED_ATLAS_DIR", "/home/coley-group/atlas/"))
    parser.add_argument("--nist-atlas-dir", default=os.environ.get("MSPRED_ATLAS_DIR_NIST", ""))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.users_file:
        print("ICEBERG_USERS_FILE or --users-file is required.", file=sys.stderr)
        return 2
    if not args.host_key:
        print("ICEBERG_SFTP_HOST_KEY or --host-key is required.", file=sys.stderr)
        return 2
    try:
        asyncio.run(_serve(args))
    except (OSError, RuntimeError) as exc:
        print(f"Failed to start ICEBERG SFTP service: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
