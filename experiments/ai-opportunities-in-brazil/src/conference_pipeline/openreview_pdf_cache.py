"""Download OpenReview PDFs into the pipeline HTTP cache with authentication."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
import threading

from .http import HttpClient
from .io import read_papers


BASE_URL = "https://api2.openreview.net"


def valid_pdf(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 4 and path.read_bytes()[:4] == b"%PDF"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint-every must be positive")

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise RuntimeError("OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD are required")

    papers = read_papers(args.input)
    cache = HttpClient(cache_dir=args.cache_dir)
    import openreview

    bootstrap_client = openreview.api.OpenReviewClient(
        baseurl=BASE_URL, username=username, password=password
    )
    token = bootstrap_client.token
    if not token:
        raise RuntimeError("OpenReview authentication did not return a session token")
    pending: list[tuple[str, Path]] = []
    cached = 0
    for paper in papers:
        if not paper.pdf_url:
            continue
        path = cache._cache_path(paper.pdf_url)
        if path is None:
            raise RuntimeError("cache-dir is required")
        if valid_pdf(path):
            cached += 1
        else:
            pending.append((paper.paper_id, path))

    state = threading.local()

    def download(item: tuple[str, Path]) -> None:
        note_id, path = item
        if not hasattr(state, "client"):
            state.client = openreview.api.OpenReviewClient(
                baseurl=BASE_URL, token=token
            )
        value = state.client.get_pdf(note_id)
        if not value.startswith(b"%PDF"):
            raise RuntimeError(f"Invalid PDF response for {note_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(value)
        temporary.replace(path)

    completed = cached
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        in_flight = set()
        for _ in range(min(len(pending), args.workers * 2)):
            in_flight.add(executor.submit(download, next(iterator)))
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
                if completed % args.checkpoint_every == 0:
                    print(f"Cached {completed}/{len(papers)} PDFs", flush=True)
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                in_flight.add(executor.submit(download, item))
    print(f"Cached {completed}/{len(papers)} PDFs", flush=True)


if __name__ == "__main__":
    main()
