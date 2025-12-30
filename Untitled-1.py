#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
每天 09:29（默认 Asia/Singapore）自动触发 getdata 拉取。
- 优先调用 getdata.main()
- 若导入失败，fallback 运行: python -m getdata
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import traceback
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

# ========= 可配置区 =========
TIMEZONE = "Asia/Singapore"
TARGET_HOUR = 9
TARGET_MINUTE = 29
TARGET_SECOND = 35

# 防止重入：上一轮还没跑完下一轮又开始（按你的需求可关闭）
USE_LOCK_FILE = True
LOCK_FILE = "auto_getdata.lock"

# 日志
LOG_FILE = "auto_getdata.log"
LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUPS = 5

# 失败重试（一次触发内）
RETRY_TIMES = 2
RETRY_SLEEP_SECONDS = 5
# ===========================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("auto_getdata")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOGGER = setup_logger()


def next_run_time(now: datetime, tz: ZoneInfo) -> datetime:
    """给定 now，计算下一个 09:29:00 的绝对时间点（含时区）。"""
    assert now.tzinfo is not None
    candidate = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=TARGET_SECOND, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate.astimezone(tz)


def acquire_lock() -> bool:
    """简单锁：存在 lock 文件则认为正在运行，避免重入。"""
    if not USE_LOCK_FILE:
        return True

    if os.path.exists(LOCK_FILE):
        return False

    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"start={datetime.now().isoformat()}\n")
        return True
    except Exception:
        return False


def release_lock() -> None:
    if not USE_LOCK_FILE:
        return
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

def run_getdata() -> None:
    """
    直接运行 getdata.py（等价于：python getdata.py）
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    getdata_path = os.path.join(base_dir, "getdata.py")

    if not os.path.exists(getdata_path):
        raise FileNotFoundError(f"Cannot find getdata.py at: {getdata_path}")

    cmd = [sys.executable, getdata_path]
    LOGGER.info(f"Running command: {' '.join(cmd)} (cwd={base_dir})")

    proc = subprocess.run(
        cmd,
        cwd=base_dir,              # 关键：保证相对路径正确
        capture_output=True,
        text=True
    )

    if proc.stdout:
        LOGGER.info(f"[getdata stdout]\n{proc.stdout.strip()}")
    if proc.stderr:
        LOGGER.warning(f"[getdata stderr]\n{proc.stderr.strip()}")

    if proc.returncode != 0:
        raise RuntimeError(f"getdata.py failed with returncode={proc.returncode}")


def main_loop() -> None:
    tz = ZoneInfo(TIMEZONE)
    LOGGER.info(f"Auto scheduler started. TZ={TIMEZONE}, target={TARGET_HOUR:02d}:{TARGET_MINUTE:02d}:{TARGET_SECOND:02d}")

    while True:
        now = datetime.now(tz)
        run_at = next_run_time(now, tz)
        sleep_seconds = (run_at - now).total_seconds()

        LOGGER.info(f"Next run at {run_at.isoformat()} (sleep {sleep_seconds:.2f}s)")
        # 睡到触发点附近
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        # 防重入锁
        if not acquire_lock():
            LOGGER.warning("Lock exists; previous run may still be running. Skip this trigger.")
            continue

        try:
            LOGGER.info("Trigger fired. Start getdata.")
            last_err: Optional[Exception] = None

            for attempt in range(RETRY_TIMES + 1):
                try:
                    run_getdata()
                    LOGGER.info("getdata finished successfully.")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    LOGGER.error(f"getdata failed (attempt {attempt+1}/{RETRY_TIMES+1}): {e}")
                    LOGGER.error(traceback.format_exc())
                    if attempt < RETRY_TIMES:
                        time.sleep(RETRY_SLEEP_SECONDS)

            if last_err is not None:
                LOGGER.error("All retries failed for this trigger.")
        finally:
            release_lock()


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user, exiting.")
