#!/usr/bin/env python3

"""Acquire or release S3 WIP locks for distributed pipeline stages."""

from __future__ import annotations

import argparse

from impresso_cookbook import setup_logging

from s3_pipeline_support import (
    acquire_wip_lock,
    build_s3_client,
    release_wip_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage S3 work-in-progress locks for distributed processing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire_parser = subparsers.add_parser(
        "acquire",
        help=(
            "Acquire a WIP lock for an S3 target. Exits 0 when acquired, 2 when the"
            " output already exists, 3 when another worker owns the lock."
        ),
    )
    acquire_parser.add_argument(
        "--s3-target",
        required=True,
        help="S3 target path to protect with a .wip file.",
        metavar="S3_PATH",
    )
    acquire_parser.add_argument(
        "--wip-max-age",
        type=float,
        default=24,
        help="Maximum age in hours before a WIP file is considered stale.",
    )
    acquire_parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Optional local or S3 files associated with the lock.",
        metavar="PATH",
    )
    acquire_parser.add_argument(
        "--local-target",
        help="Optional local target path that this lock acquisition guards.",
        metavar="PATH",
    )

    release_parser = subparsers.add_parser(
        "release",
        help="Release the .wip file for an S3 target if present.",
    )
    release_parser.add_argument(
        "--s3-target",
        required=True,
        help="S3 target path whose .wip file should be removed.",
        metavar="S3_PATH",
    )

    for subparser in [acquire_parser, release_parser]:
        subparser.add_argument(
            "--log-file",
            dest="log_file",
            help="Write log to FILE",
            metavar="FILE",
        )
        subparser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: %(default)s)",
        )

    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    s3_client = build_s3_client()

    if args.command == "acquire":
        raise SystemExit(
            acquire_wip_lock(
                s3_client,
                args.s3_target,
                args.wip_max_age,
                files=args.files,
                local_target=args.local_target,
            )
        )

    raise SystemExit(release_wip_lock(s3_client, args.s3_target))


if __name__ == "__main__":
    main()
