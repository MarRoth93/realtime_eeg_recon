"""Mock LSL marker streamer for trigger-driven testing."""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Sequence


def create_marker_outlet(
    name: str = "TaskMarkers",
    stream_type: str = "Markers",
    source_id: str = "maryam_mock_markers",
) -> object:
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(name, stream_type, 1, 0.0, "string", source_id)
    return StreamOutlet(info, chunk_size=1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit periodic mock task markers over LSL.")
    parser.add_argument("--stream-name", default="TaskMarkers", help="LSL marker stream name.")
    parser.add_argument("--marker", default="stim_onset", help="Marker value to emit.")
    parser.add_argument("--image-id", default=None, help="Optional image id to embed in the marker payload.")
    parser.add_argument("--image-path", default=None, help="Optional image path to embed in the marker payload.")
    parser.add_argument(
        "--payload-json",
        default=None,
        help="Optional raw JSON marker payload. Overrides --marker/--image-id/--image-path.",
    )
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Seconds between markers.")
    parser.add_argument("--count", type=int, default=0, help="Number of markers to send. 0 means infinite.")
    parser.add_argument("--start-delay-seconds", type=float, default=1.0, help="Delay before first marker.")
    return parser.parse_args(argv)


def _payload_for_args(args: argparse.Namespace) -> str:
    if args.payload_json:
        return str(args.payload_json)
    if args.image_id or args.image_path:
        return json.dumps(
            {
                "event": args.marker,
                "image_id": args.image_id,
                "image_path": args.image_path,
            }
        )
    return str(args.marker)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outlet = create_marker_outlet(name=args.stream_name)
    payload = _payload_for_args(args)
    print(
        f"Streaming marker '{payload}' on '{args.stream_name}' every "
        f"{args.interval_seconds:.2f}s. Press Ctrl+C to stop."
    )
    time.sleep(args.start_delay_seconds)

    sent = 0
    try:
        while args.count == 0 or sent < args.count:
            outlet.push_sample([payload])
            sent += 1
            print(f"sent marker {sent}: {payload}")
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nMarker streaming stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
