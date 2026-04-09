#!/usr/bin/env python3

"""Shared S3 readiness and WIP helpers for distributed pipeline stages."""

from __future__ import annotations

import getpass
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Optional

from botocore.exceptions import ClientError
from dotenv import load_dotenv

from impresso_cookbook import get_s3_client, parse_s3_path, s3_file_exists

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_NOT_READY = 1
EXIT_OUTPUT_EXISTS = 2
EXIT_LOCKED = 3

load_dotenv()


def build_s3_client() -> Any:
    """Return the configured S3 client."""
    return get_s3_client()


def warn_skipped_build_attempt(
    reason: str,
    s3_target: str,
    local_target: Optional[str] = None,
) -> None:
    """Warn that the current invocation did not build the requested target."""
    if local_target:
        log.warning(
            "Skipping build attempt for %s (%s); local target %s was not built in"
            " this invocation",
            s3_target,
            reason,
            local_target,
        )
    else:
        log.warning(
            "Skipping build attempt for %s (%s); no local artifact was built in"
            " this invocation",
            s3_target,
            reason,
        )


def _read_wip_object(s3_client: Any, wip_path: str) -> Optional[dict]:
    """Return WIP metadata if the WIP object exists."""
    bucket, key = parse_s3_path(wip_path)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise

    body = response["Body"].read().decode("utf-8")
    if not body.strip():
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _describe_wip_owner(s3_client: Any, wip_path: str) -> str:
    """Return a compact description of the active WIP owner."""
    info = _read_wip_object(s3_client, wip_path) or {}
    username = info.get("username", "unknown")
    hostname = info.get("hostname", "unknown")
    ip_address = info.get("ip_address", "unknown")
    return f"user={username} host={hostname} ip={ip_address}"


def _get_wip_age_hours(s3_client: Any, wip_path: str) -> Optional[float]:
    """Return WIP age in hours if the WIP object exists."""
    bucket, key = parse_s3_path(wip_path)

    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise

    last_modified = response["LastModified"]
    now = datetime.now(timezone.utc)
    return (now - last_modified).total_seconds() / 3600


def clear_stale_wip_if_needed(s3_client: Any, s3_path: str, wip_max_age: float) -> bool:
    """Remove a stale WIP file if it exceeds the configured age."""
    wip_path = s3_path + ".wip"
    age_hours = _get_wip_age_hours(s3_client, wip_path)
    if age_hours is None:
        return False

    if age_hours <= wip_max_age:
        return False

    bucket, key = parse_s3_path(wip_path)
    log.info(
        "Removing stale WIP file %.1f hours old: %s",
        age_hours,
        wip_path,
    )
    s3_client.delete_object(Bucket=bucket, Key=key)
    return True


def has_active_wip(s3_client: Any, s3_path: str, wip_max_age: float) -> bool:
    """Return whether a non-stale WIP file currently exists for the target."""
    clear_stale_wip_if_needed(s3_client, s3_path, wip_max_age)
    wip_path = s3_path + ".wip"
    age_hours = _get_wip_age_hours(s3_client, wip_path)
    if age_hours is None:
        return False

    owner = _describe_wip_owner(s3_client, wip_path)
    log.info(
        "Active WIP %.1f hours old for %s (%s)",
        age_hours,
        wip_path,
        owner,
    )
    return True


def is_target_ready(s3_client: Any, s3_path: str, wip_max_age: float) -> bool:
    """Return whether a target exists on S3 and is not currently locked."""
    if not s3_file_exists(s3_client, s3_path):
        log.info("Required S3 output missing: %s", s3_path)
        return False

    if has_active_wip(s3_client, s3_path, wip_max_age):
        log.info("Required S3 output is still locked: %s", s3_path)
        return False

    return True


def acquire_wip_lock(
    s3_client: Any,
    s3_path: str,
    wip_max_age: float,
    files: Optional[list[str]] = None,
    local_target: Optional[str] = None,
    force: bool = False,
) -> int:
    """Acquire a target WIP lock, or return a skip code if unavailable.

    Args:
        s3_client: Configured S3 client
        s3_path: S3 target path to protect with WIP lock
        wip_max_age: Maximum age in hours for stale WIP detection
        files: Optional list of associated files
        local_target: Optional local target path for warning messages
        force: If True, skip the "output already exists" check (for force-overwrite scenarios)

    Returns:
        EXIT_OK (0) if lock acquired
        EXIT_OUTPUT_EXISTS (2) if output exists and force=False
        EXIT_LOCKED (3) if another worker has the lock
    """
    if not force and s3_file_exists(s3_client, s3_path):
        log.info("S3 output already exists: %s", s3_path)
        warn_skipped_build_attempt("output already exists on S3", s3_path, local_target)
        return EXIT_OUTPUT_EXISTS

    if has_active_wip(s3_client, s3_path, wip_max_age):
        warn_skipped_build_attempt("active WIP lock exists", s3_path, local_target)
        return EXIT_LOCKED

    hostname = socket.gethostname()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        sock.close()
    except OSError:
        ip_address = "127.0.0.1"

    wip_info = {
        "hostname": hostname,
        "ip_address": ip_address,
        "username": getpass.getuser(),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "target": s3_path,
        "files": files or [],
    }

    wip_path = s3_path + ".wip"
    bucket, key = parse_s3_path(wip_path)

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(wip_info, indent=2).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        log.info("Created WIP file: %s", wip_path)
        return EXIT_OK
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"PreconditionFailed", "ConditionalRequestConflict"}:
            log.info("Another worker acquired the WIP first for %s", s3_path)
            warn_skipped_build_attempt(
                "another worker acquired the WIP first",
                s3_path,
                local_target,
            )
            return EXIT_LOCKED
        raise


def release_wip_lock(s3_client: Any, s3_path: str) -> int:
    """Remove a target WIP lock if present."""
    wip_path = s3_path + ".wip"
    bucket, key = parse_s3_path(wip_path)

    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
        log.info("Removed WIP file: %s", wip_path)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code not in {"404", "NoSuchKey", "NotFound"}:
            raise

    return EXIT_OK
