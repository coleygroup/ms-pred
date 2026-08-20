"""Read-only virtual filesystem for the ICEBERG atlas exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable


NIST_ROLES = {"admin", "authorized_user"}
HIDDEN_SUFFIXES = (".idx",)


class VfsError(Exception):
    """Base class for virtual filesystem errors."""


class VfsPermissionError(PermissionError, VfsError):
    """Raised when a virtual path is not visible to the current user."""


class VfsNotFoundError(FileNotFoundError, VfsError):
    """Raised when a virtual path does not exist."""


class VfsNotDirectoryError(NotADirectoryError, VfsError):
    """Raised when a virtual path is not a directory."""


@dataclass(frozen=True)
class VfsEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: float | None = None

    @property
    def mode(self) -> int:
        base = stat.S_IFDIR if self.is_dir else stat.S_IFREG
        perms = 0o555 if self.is_dir else 0o444
        return base | perms


@dataclass(frozen=True)
class _Mount:
    virtual_root: PurePosixPath
    real_root: Path
    label: str

    @property
    def real_root_resolved(self) -> Path:
        return self.real_root.resolve(strict=False)


def role_can_access_nist(role: str, email: str = "", admin_emails: Iterable[str] = ()) -> bool:
    normalized_admins = {item.strip().lower() for item in admin_emails if item.strip()}
    return role in NIST_ROLES or email.strip().lower() in normalized_admins


def normalize_virtual_path(path: str | bytes | PurePosixPath) -> PurePosixPath:
    """Normalize an SFTP/WebUI path and reject traversal."""
    if isinstance(path, bytes):
        path = path.decode("utf-8", errors="strict")
    path_str = str(path or "/").replace("\\", "/")
    if not path_str.startswith("/"):
        path_str = "/" + path_str

    parts: list[str] = []
    for part in PurePosixPath(path_str).parts:
        if part in ("", "/", "."):
            continue
        if part == "..":
            if not parts:
                raise VfsPermissionError("Path traversal is not allowed")
            parts.pop()
            continue
        parts.append(part)
    return PurePosixPath("/" + "/".join(parts))


class AtlasVirtualFS:
    """Role-aware read-only view over public and gated atlas MGF trees."""

    def __init__(self, public_atlas_dir: Path | str, nist_atlas_dir: Path | str | None = None):
        self.public_atlas_dir = Path(public_atlas_dir)
        self.nist_atlas_dir = Path(nist_atlas_dir) if nist_atlas_dir else None

    def _all_mounts(self, include_nist: bool) -> list[_Mount]:
        mounts = [
            _Mount(PurePosixPath("/public/h_plus"), self.public_atlas_dir / "h_plus_out_mgf", "Positive mode"),
            _Mount(PurePosixPath("/public/h_minus"), self.public_atlas_dir / "h_minus_out_mgf", "Negative mode"),
        ]
        if include_nist and self.nist_atlas_dir is not None:
            mounts.extend(
                [
                    _Mount(PurePosixPath("/nist23/h_plus"), self.nist_atlas_dir / "h_plus_out_mgf", "NIST'23 positive mode"),
                    _Mount(PurePosixPath("/nist23/h_minus"), self.nist_atlas_dir / "h_minus_out_mgf", "NIST'23 negative mode"),
                ]
            )
        return [mount for mount in mounts if mount.real_root_resolved.is_dir()]

    def visible_root_names(self, include_nist: bool) -> list[str]:
        names = ["public"]
        if include_nist and self.nist_atlas_dir is not None:
            names.append("nist23")
        return names

    def _find_mount(self, vpath: PurePosixPath, include_nist: bool) -> tuple[_Mount, PurePosixPath] | None:
        for mount in self._all_mounts(include_nist):
            if vpath == mount.virtual_root:
                return mount, PurePosixPath(".")
            try:
                rel = vpath.relative_to(mount.virtual_root)
            except ValueError:
                continue
            return mount, rel
        return None

    def _check_category_access(self, vpath: PurePosixPath, include_nist: bool) -> None:
        parts = [p for p in vpath.parts if p != "/"]
        if parts and parts[0] == "nist23" and "nist23" not in self.visible_root_names(include_nist):
            raise VfsPermissionError("NIST'23 atlas access is not permitted for this user")
        if parts and parts[0] not in self.visible_root_names(include_nist):
            raise VfsNotFoundError(str(vpath))

    @staticmethod
    def _is_inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_hidden_path(path: PurePosixPath) -> bool:
        return any(part.endswith(HIDDEN_SUFFIXES) for part in path.parts if part != "/")

    def resolve_real_path(self, path: str | bytes | PurePosixPath, include_nist: bool) -> Path:
        vpath = normalize_virtual_path(path)
        if self._is_hidden_path(vpath):
            raise VfsNotFoundError(str(vpath))
        self._check_category_access(vpath, include_nist)
        match = self._find_mount(vpath, include_nist)
        if match is None:
            raise VfsNotFoundError(str(vpath))
        mount, rel = match
        candidate = mount.real_root if str(rel) == "." else mount.real_root / Path(*rel.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise VfsNotFoundError(str(vpath)) from exc
        root = mount.real_root_resolved
        if resolved != root and not self._is_inside(resolved, root):
            raise VfsPermissionError("Resolved path escapes the atlas root")
        return resolved

    def stat(self, path: str | bytes | PurePosixPath, include_nist: bool) -> VfsEntry:
        vpath = normalize_virtual_path(path)
        self._check_category_access(vpath, include_nist)
        if vpath == PurePosixPath("/"):
            return VfsEntry("", "/", True)

        parts = [p for p in vpath.parts if p != "/"]
        if len(parts) == 1 and parts[0] in self.visible_root_names(include_nist):
            return VfsEntry(parts[0], str(vpath), True)

        if len(parts) == 2 and self._find_mount(vpath, include_nist) is not None:
            return VfsEntry(parts[-1], str(vpath), True)

        real = self.resolve_real_path(vpath, include_nist)
        st = real.stat()
        return VfsEntry(vpath.name, str(vpath), real.is_dir(), st.st_size, st.st_mtime)

    def listdir(self, path: str | bytes | PurePosixPath, include_nist: bool) -> list[VfsEntry]:
        vpath = normalize_virtual_path(path)
        self._check_category_access(vpath, include_nist)

        if vpath == PurePosixPath("/"):
            return [VfsEntry(name, f"/{name}", True) for name in self.visible_root_names(include_nist)]

        parts = [p for p in vpath.parts if p != "/"]
        if len(parts) == 1 and parts[0] in self.visible_root_names(include_nist):
            prefix = f"/{parts[0]}"
            return [
                VfsEntry(mount.virtual_root.name, str(mount.virtual_root), True)
                for mount in self._all_mounts(include_nist)
                if str(mount.virtual_root).startswith(prefix + "/")
            ]

        real_dir = self.resolve_real_path(vpath, include_nist)
        if not real_dir.is_dir():
            raise VfsNotDirectoryError(str(vpath))

        entries: list[VfsEntry] = []
        for child in real_dir.iterdir():
            if child.name.endswith(HIDDEN_SUFFIXES):
                continue
            try:
                resolved_child = child.resolve(strict=True)
            except FileNotFoundError:
                continue
            mount_match = self._find_mount(vpath, include_nist)
            if mount_match is None:
                continue
            mount, _ = mount_match
            root = mount.real_root_resolved
            if resolved_child != root and not self._is_inside(resolved_child, root):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(
                VfsEntry(
                    child.name,
                    str(vpath / child.name),
                    child.is_dir(),
                    st.st_size,
                    st.st_mtime,
                )
            )
        entries.sort(key=lambda item: (not item.is_dir, item.name.lower()))
        return entries

    def open_read(self, path: str | bytes | PurePosixPath, include_nist: bool):
        real = self.resolve_real_path(path, include_nist)
        if not real.is_file():
            raise VfsNotFoundError(str(path))
        return real.open("rb")

    def deny_write(self, *_args, **_kwargs) -> None:
        raise VfsPermissionError("The ICEBERG atlas SFTP service is read-only")
