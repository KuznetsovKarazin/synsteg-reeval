"""Shared paths for ported audit steps; never imports remote/API clients."""
import argparse
from pathlib import Path


def parse_paths(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source', required=True, type=Path,
                        help='Extracted original synsteg_reproduction directory')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output directory for this step')
    parser.add_argument('--source-archive', type=Path,
                        help='Optional original ZIP for provenance')
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / 'reports').is_dir():
        parser.error('--source must contain reports/')
    if args.source_archive and not args.source_archive.is_file():
        parser.error('--source-archive does not exist')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    return source, output, args.source_archive
