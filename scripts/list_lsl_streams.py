#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _channel_labels(info) -> list[str]:
    labels: list[str] = []
    try:
        channels = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            label = channels.child_value("label")
            if label:
                labels.append(label)
            channels = channels.next_sibling()
    except Exception:
        return labels
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="List visible Lab Streaming Layer streams.")
    parser.add_argument("--wait-time", type=float, default=3.0, help="Seconds to wait for stream discovery.")
    args = parser.parse_args()

    try:
        from pylsl import resolve_streams
    except Exception as exc:
        raise SystemExit(f"pylsl is required to list LSL streams: {exc}") from exc

    streams = resolve_streams(wait_time=args.wait_time)
    if not streams:
        print(f"No LSL streams found after {args.wait_time:g} seconds.")
        return 1

    for idx, info in enumerate(streams, start=1):
        print(f"[{idx}] name={info.name()!r}")
        print(f"    type={info.type()!r}")
        print(f"    source_id={info.source_id()!r}")
        print(f"    channels={info.channel_count()}")
        print(f"    nominal_srate={info.nominal_srate():g}")
        print(f"    channel_format={info.channel_format()}")
        labels = _channel_labels(info)
        if labels:
            print(f"    labels={', '.join(labels)}")
        else:
            print("    labels=<not exposed in stream metadata>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
