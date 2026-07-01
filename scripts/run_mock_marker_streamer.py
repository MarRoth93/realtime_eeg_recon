#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main() -> int:
    from maryam_rt.realtime.streaming.mock_marker_streamer import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
