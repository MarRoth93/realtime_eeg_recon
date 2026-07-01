from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from maryam_rt.integration.marker_payload import ParsedMarkerPayload


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class ResolvedTarget:
    image_path: Path
    image_id: str | None = None
    label: str | None = None


class ImageTargetResolver:
    """Resolve target images from marker payload metadata."""

    def __init__(self, image_root: str | Path | None = None) -> None:
        self.image_root = None if image_root is None else Path(image_root)
        self._index: dict[str, Path] = {}
        if self.image_root is not None:
            self._index = self._build_index(self.image_root)

    def resolve(self, payload: ParsedMarkerPayload) -> ResolvedTarget | None:
        if payload.image_path:
            image_path = Path(payload.image_path).expanduser()
            if image_path.exists():
                return ResolvedTarget(image_path=image_path, image_id=payload.image_id, label=self._label_for(image_path))

        if payload.image_id is None or not self._index:
            return None

        candidates = [
            payload.image_id,
            payload.image_id.lower(),
        ]
        digits = "".join(ch for ch in payload.image_id if ch.isdigit())
        if digits:
            candidates.extend({digits, digits.zfill(5)})

        for key in candidates:
            path = self._index.get(key)
            if path is not None and path.exists():
                return ResolvedTarget(image_path=path, image_id=payload.image_id, label=self._label_for(path))
        return None

    def _build_index(self, root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            parent_name = path.parent.name
            file_stem = path.stem
            for key in {
                parent_name,
                parent_name.lower(),
                file_stem,
                file_stem.lower(),
            }:
                index.setdefault(key, path)

            if "_" in parent_name:
                prefix = parent_name.split("_", 1)[0]
                index.setdefault(prefix, path)
                index.setdefault(prefix.lower(), path)
        return index

    def _label_for(self, image_path: Path) -> str:
        parent_name = image_path.parent.name
        if "_" in parent_name:
            return parent_name.split("_", 1)[1]
        return parent_name

