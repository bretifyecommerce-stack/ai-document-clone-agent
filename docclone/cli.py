from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import replicate_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document clone agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    replicate = sub.add_parser("replicate", help="Replicate document images into a PDF")
    replicate.add_argument("--input", required=True, help="Input image or directory")
    replicate.add_argument("--output", required=True, help="Output PDF path")
    replicate.add_argument("--config", help="Optional config path (json/yaml)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "replicate":
        output = replicate_images(args.input, args.output, args.config)
        print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()
