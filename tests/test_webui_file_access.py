import importlib
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml
from werkzeug.security import generate_password_hash

from webui.vfs import AtlasVirtualFS, VfsNotFoundError, VfsPermissionError
from webui.sftp_server import check_sftp_credentials, log_sftp_request, role_for_email


def _write_users(path: Path, password: str = "Password!1") -> None:
    users = {
        "plain@example.com": {
            "password": generate_password_hash(password),
            "role": "user",
            "created": "",
        },
        "auth@example.com": {
            "password": generate_password_hash(password),
            "role": "authorized_user",
            "created": "",
        },
        "admin@example.com": {
            "password": generate_password_hash(password),
            "role": "admin",
            "created": "",
        },
    }
    path.write_text(yaml.safe_dump(users), encoding="utf-8")


def _make_atlas(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    public = tmp_path / "atlas"
    nist = tmp_path / "atlas_nist"
    public_h = public / "h_plus_out_mgf"
    public_m = public / "h_minus_out_mgf"
    nist_h = nist / "h_plus_out_mgf"
    nist_m = nist / "h_minus_out_mgf"
    for path in (public_h, public_m, nist_h, nist_m):
        path.mkdir(parents=True)

    public_bytes = b"BEGIN IONS\nFORMULA=C2H6O\nEND IONS\n"
    nist_bytes = b"BEGIN IONS\nFORMULA=C3H8O\nEND IONS\n"
    (public_h / "public.mgf").write_bytes(public_bytes)
    (public_h / "public.mgf.idx").write_bytes(b"hidden index")
    (nist_h / "nist.mgf").write_bytes(nist_bytes)
    return public, nist, public_bytes, nist_bytes


def test_vfs_role_visibility_and_path_safety(tmp_path):
    public, nist, public_bytes, _ = _make_atlas(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (public / "h_plus_out_mgf" / "escape.txt").symlink_to(outside)

    vfs = AtlasVirtualFS(public, nist)

    assert [entry.name for entry in vfs.listdir("/", include_nist=False)] == ["public"]
    assert [entry.name for entry in vfs.listdir("/", include_nist=True)] == ["public", "nist23"]
    assert vfs.open_read("/public/h_plus/public.mgf", include_nist=False).read() == public_bytes

    with pytest.raises(VfsPermissionError):
        vfs.listdir("/nist23", include_nist=False)
    with pytest.raises(VfsPermissionError):
        vfs.resolve_real_path("/../public/h_plus/public.mgf", include_nist=False)
    with pytest.raises(VfsPermissionError):
        vfs.resolve_real_path("/public/h_plus/escape.txt", include_nist=False)
    with pytest.raises(VfsPermissionError):
        vfs.deny_write()

    names = [entry.name for entry in vfs.listdir("/public/h_plus", include_nist=False)]
    assert "escape.txt" not in names
    assert "public.mgf.idx" not in names
    with pytest.raises(VfsNotFoundError):
        vfs.open_read("/public/h_plus/public.mgf.idx", include_nist=False)


def test_sftp_uses_webui_user_records(tmp_path):
    users_file = tmp_path / "users.yaml"
    _write_users(users_file)

    assert check_sftp_credentials(users_file, "auth@example.com", "Password!1")
    assert not check_sftp_credentials(users_file, "auth@example.com", "wrong")
    assert role_for_email(users_file, "plain@example.com", set()) == "user"
    assert role_for_email(users_file, "auth@example.com", set()) == "authorized_user"
    assert role_for_email(users_file, "override@example.com", {"override@example.com"}) == "admin"


def test_sftp_analytics_logs_to_requests_source(monkeypatch, tmp_path):
    db_path = tmp_path / "analytics.db"
    monkeypatch.setenv("ICEBERG_ANALYTICS_DB", str(db_path))

    log_sftp_request(None, "auth@example.com", "203.0.113.10", "listdir", "/public/h_plus")

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT ip, path, method, user_id, source FROM requests"
        ).fetchone()
        cols = {item[1] for item in conn.execute("PRAGMA table_info(requests)").fetchall()}

    assert row == (
        "203.0.113.10",
        "/sftp/listdir/public/h_plus",
        "SFTP",
        "auth@example.com",
        "sftp",
    )
    assert "source" in cols


def _load_webui_app(monkeypatch, tmp_path: Path, public: Path, nist: Path, users_file: Path):
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret")
    monkeypatch.setenv("MSPRED_ATLAS_DIR", str(public))
    monkeypatch.setenv("MSPRED_ATLAS_DIR_NIST", str(nist))
    monkeypatch.setenv("ICEBERG_USERS_FILE", str(users_file))
    monkeypatch.setenv("ICEBERG_ANALYTICS_DB", str(tmp_path / "analytics.db"))
    sys.modules.pop("webui.app", None)
    module = importlib.import_module("webui.app")
    module.app.config.update(TESTING=True)
    return module.app


def test_files_route_requires_login_and_filters_nist(monkeypatch, tmp_path):
    public, nist, _, _ = _make_atlas(tmp_path)
    users_file = tmp_path / "users.yaml"
    _write_users(users_file)
    app = _load_webui_app(monkeypatch, tmp_path, public, nist, users_file)

    client = app.test_client()
    anon = client.get("/files")
    assert anon.status_code == 302
    assert "/login" in anon.headers["Location"]

    login = client.post(
        "/login",
        data={"email": "plain@example.com", "password": "Password!1"},
    )
    assert login.status_code == 302

    root = client.get("/files")
    assert root.status_code == 200
    assert b"public/" in root.data
    assert b"nist23" not in root.data
    assert client.get("/files/nist23").status_code == 403


def test_files_authorized_user_can_browse_and_download_nist(monkeypatch, tmp_path):
    public, nist, _, nist_bytes = _make_atlas(tmp_path)
    users_file = tmp_path / "users.yaml"
    _write_users(users_file)
    app = _load_webui_app(monkeypatch, tmp_path, public, nist, users_file)

    client = app.test_client()
    login = client.post(
        "/login",
        data={"email": "auth@example.com", "password": "Password!1"},
    )
    assert login.status_code == 302

    root = client.get("/files")
    assert root.status_code == 200
    assert b"public/" in root.data
    assert b"nist23/" in root.data

    listing = client.get("/files/nist23/h_plus")
    assert listing.status_code == 200
    assert b"nist.mgf" in listing.data

    download = client.get("/files/download/nist23/h_plus/nist.mgf")
    assert download.status_code == 200
    assert download.data == nist_bytes
