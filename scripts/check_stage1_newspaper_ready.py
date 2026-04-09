#!/usr/bin/env python3

"""Check whether all stage-1 outputs for a newspaper are complete on S3."""

from __future__ import annotations

import argparse

from impresso_cookbook import setup_logging

from s3_pipeline_support import (
    EXIT_NOT_READY,
    EXIT_OK,
    build_s3_client,
    is_target_ready,
    warn_skipped_build_attempt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that all expected stage-1 outputs exist on S3 and are not locked"
            " by active WIP files."
        )
    )
    parser.add_argument(
        "--stage1-output",
        action="append",
        required=True,
        help="Expected stage-1 S3 output path. Repeat for each yearly output.",
        metavar="S3_PATH",
    )
    parser.add_argument(
        "--wip-max-age",
        type=float,
        default=24,
        help="Maximum age in hours before a WIP file is considered stale.",
    )
    parser.add_argument(
        "--local-target",
        help="Optional local target path that this readiness check guards.",
        metavar="PATH",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        help="Write log to FILE",
        metavar="FILE",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    s3_client = build_s3_client()

    for s3_path in args.stage1_output:
        if not is_target_ready(s3_client, s3_path, args.wip_max_age):
            warn_skipped_build_attempt(
                "stage-1 newspaper readiness not yet satisfied",
                s3_path,
                args.local_target,
            )
            raise SystemExit(EXIT_NOT_READY)

    raise SystemExit(EXIT_OK)


if __name__ == "__main__":
    main()
