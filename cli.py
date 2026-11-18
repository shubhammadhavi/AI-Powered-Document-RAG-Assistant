#!/usr/bin/env python3
"""Use the assistant from the terminal, without starting the server.

    python cli.py add docs/handbook.pdf
    python cli.py ask "how do I rotate the API keys?"
    python cli.py list
    python cli.py status
    python cli.py reset
"""

from __future__ import annotations

import argparse
import sys

from app.engine import get_engine
from app.loaders import EmptyDocument, UnsupportedFile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Document RAG Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="index one or more documents")
    add.add_argument("paths", nargs="+")

    ask = sub.add_parser("ask", help="ask a question about the indexed documents")
    ask.add_argument("question", nargs="+")
    ask.add_argument("-k", "--top-k", type=int, default=None)

    sub.add_parser("list", help="list indexed documents")
    sub.add_parser("status", help="show which retriever and model are in use")
    sub.add_parser("reset", help="delete the whole index")

    args = parser.parse_args(argv)
    engine = get_engine()

    if args.command == "add":
        failed = False
        for path in args.paths:
            try:
                result = engine.ingest_file(path)
            except (UnsupportedFile, EmptyDocument) as exc:
                print(f"skipped {path}: {exc}", file=sys.stderr)
                failed = True
                continue
            print(
                f"indexed {result['source']}: {result['chunks_indexed']} chunks "
                f"from {result['pages']} page(s)"
            )
        return 1 if failed else 0

    if args.command == "ask":
        result = engine.ask(" ".join(args.question), args.top_k)
        print(result.answer)
        if result.sources:
            print("\nSources: " + ", ".join(result.sources))
        print(f"[{result.provider} / {result.model}]")
        return 0

    if args.command == "list":
        documents = engine.documents()
        if not documents:
            print("No documents indexed yet.")
        for name, count in documents.items():
            print(f"{count:>5} chunks  {name}")
        return 0

    if args.command == "status":
        for key, value in engine.status().items():
            print(f"{key:>16}: {value}")
        return 0

    if args.command == "reset":
        print(f"removed {engine.reset()} chunks")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
