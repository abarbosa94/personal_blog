"""Correct a known venue-key provenance error in paper or reconciled JSONL."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .io import (
    read_papers,
    read_reconciled,
    write_jsonl,
    write_reconciled_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--from-venue", required=True)
    parser.add_argument("--to-venue", required=True)
    parser.add_argument("--kind", choices=("paper", "reconciled"), required=True)
    args = parser.parse_args()
    if args.kind == "paper":
        records = read_papers(args.input)
        output = [
            replace(record, venue_key=args.to_venue)
            if record.venue_key == args.from_venue
            else record
            for record in records
        ]
        count = write_jsonl(output, args.output)
    else:
        records = read_reconciled(args.input)
        output = [
            replace(
                record,
                paper=replace(record.paper, venue_key=args.to_venue),
            )
            if record.paper.venue_key == args.from_venue
            else record
            for record in records
        ]
        count = write_reconciled_jsonl(output, args.output)
    changed = sum(
        record.paper.venue_key == args.to_venue
        if args.kind == "reconciled"
        else record.venue_key == args.to_venue
        for record in output
    )
    print(f"Wrote {count} records; {changed} carry venue {args.to_venue}")


if __name__ == "__main__":
    main()
