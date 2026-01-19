#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GetData 智能调度器

功能：
1. 自动检测下一个有效交易时段(A股+港股都开市）
2. 等待到时间后自动启动 getdata 数据采集
3. 在非交易时间暂停采集，交易时间恢复

时间窗口：
- T1: 09:29:50 ~ 11:30:10 (早盘)
- T2: 12:00:00 ~ 12:00:30 (午间快照)
- T3: 12:59:50 ~ 15:00:10 (午盘)
"""

from __future__ import annotations

import os
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple
import aiohttp

# ========= 配置区 =========
TIMEZONE = "Asia/Shanghai"
TZ = ZoneInfo(TIMEZONE)

# 采集时间窗口
TRADING_WINDOWS = [
    (dt_time(9, 29, 25), dt_time(11, 30, 10)),  # T1: 早盘
    (
        dt_time(12, 0, 0),
        dt_time(12, 0, 30),
    ),  # T2: 午间快照 (A股11:30收 H股12:00收 捕捉套利窗口)
    (dt_time(12, 59, 25), dt_time(15, 0, 10)),  # T3: 午盘
]

# 日志配置
LOG_FILE = "logs/scheduler.log"
LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUPS = 5


# ========= 日志设置 =========
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOGGER = setup_logger()


# ========= 交易日历检测 =========
class TradingCalendar:
    """
    交易日历检测器
    使用 exchange_calendars 库检查 A股(XSHG) 和 港股(XHKG) 是否都是交易日
    """

    def __init__(self):
        try:
            import exchange_calendars as ecals
            import pandas as pd

            self.xshg = ecals.get_calendar("XSHG")  # 上交所
            self.xhkg = ecals.get_calendar("XHKG")  # 港交所
            self.pd = pd
            self._has_lib = True
            LOGGER.info("成功加载 exchange_calendars (XSHG/XHKG)")
        except ImportError:
            self._has_lib = False
            LOGGER.warning("未安装 exchange_calendars，将回退到简单模式")

    async def is_a_share_trading_day(self, target_date: date) -> bool:
        """检查 A股 是否交易日"""
        if not self._has_lib:
            return target_date.weekday() < 5

        ts = self.pd.Timestamp(target_date)
        return self.xshg.is_session(ts)

    async def is_hk_trading_day(self, target_date: date) -> bool:
        """检查 港股 是否交易日"""
        if not self._has_lib:
            return target_date.weekday() < 5

        ts = self.pd.Timestamp(target_date)
        return self.xhkg.is_session(ts)

    async def is_both_trading(self, target_date: date) -> bool:
        """检查 A股 和 港股 是否都交易"""
        a_share = await self.is_a_share_trading_day(target_date)
        hk = await self.is_hk_trading_day(target_date)

        status = "开市" if (a_share and hk) else "休市"
        LOGGER.info(
            f"交易日检测 {target_date}: A股={'开' if a_share else '休'} / 港股={'开' if hk else '休'} -> {status}"
        )

        return a_share and hk


# ========= 时间窗口计算 =========
def is_in_trading_window(now: datetime) -> bool:
    """检查当前时间是否在交易窗口内"""
    current_time = now.time()
    for start, end in TRADING_WINDOWS:
        if start <= current_time <= end:
            return True
    return False


def get_next_window_start(now: datetime) -> Tuple[datetime, str]:
    """
    计算下一个交易窗口的开始时间
    返回: (下一个窗口开始的 datetime, 窗口名称)
    """
    current_time = now.time()
    today = now.date()

    # 检查今天剩余的窗口
    for i, (start, end) in enumerate(TRADING_WINDOWS):
        if current_time < start:
            # 今天还有窗口没开始
            window_start = datetime.combine(today, start, tzinfo=TZ)
            return window_start, f"T{i+1}"
        elif start <= current_time <= end:
            # 当前在窗口内
            return now, f"T{i+1} (当前)"

    # 今天的窗口都结束了，返回明天的第一个窗口
    tomorrow = today + timedelta(days=1)
    first_start = TRADING_WINDOWS[0][0]
    window_start = datetime.combine(tomorrow, first_start, tzinfo=TZ)
    return window_start, "T1 (明日)"


def get_current_window_end(now: datetime) -> Optional[datetime]:
    """获取当前窗口的结束时间"""
    current_time = now.time()
    today = now.date()

    for start, end in TRADING_WINDOWS:
        if start <= current_time <= end:
            return datetime.combine(today, end, tzinfo=TZ)

    return None


# ========= 主调度器 =========
class Scheduler:
    """
    智能调度器

    工作流程：
    1. 检查下一个交易日（A股+港股都开市）
    2. 等待到交易窗口开始
    3. 启动 getdata_v2 采集
    4. 窗口结束后暂停，等待下一个窗口
    """

    def __init__(self):
        self.calendar = TradingCalendar()
        self.running = False
        self.collector_task: Optional[asyncio.Task] = None

    async def find_next_trading_day(self, start_date: date) -> date:
        """找到下一个 A股+港股 都交易的日期"""
        current = start_date
        max_days = 30  # 最多查 30 天

        for _ in range(max_days):
            if await self.calendar.is_both_trading(current):
                return current
            current += timedelta(days=1)

        # 默认返回下一个工作日
        LOGGER.warning("未能确定交易日，使用下一个工作日")
        while current.weekday() >= 5:
            current += timedelta(days=1)
        return current

    async def wait_until(self, target: datetime):
        """等待到指定时间"""
        now = datetime.now(TZ)
        wait_seconds = (target - now).total_seconds()

        if wait_seconds > 0:
            LOGGER.info(
                f"等待 {wait_seconds:.0f} 秒 ({wait_seconds/3600:.2f} 小时) 到 {target.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await asyncio.sleep(wait_seconds)

    async def start_collector(self):
        """启动数据采集模块"""
        LOGGER.info("启动 getdata_v2 数据采集...")

        # 优先尝试直接导入运行 (在同一进程中)
        try:
            from getdata.main import main as getdata_main

            # 注意: 如果 getdata.main 没有处理 CancelledError，这里取消时可能会报错
            # 但既然是同一进程，asyncio.CancelledError 会在 await 处抛出
            await getdata_main()
            return
        except ImportError:
            pass
        except asyncio.CancelledError:
            LOGGER.info("采集任务已取消 (Import模式)")
            raise
        except Exception as e:
            LOGGER.error(f"Import模式运行失败: {e}，尝试 Subprocess 模式")

        # 备用方案：异步 Subprocess
        LOGGER.info("切换至 Subprocess 模式启动...")
        import sys

        cmd = [sys.executable, "-m", "getdata.main"]
        base_dir = os.path.dirname(os.path.abspath(__file__))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=base_dir,
            # 不需要 pipe stdout/stderr，直接输出到当前终端
        )

        try:
            await process.wait()
        except asyncio.CancelledError:
            LOGGER.info("采集任务取消，正在终止子进程...")
            process.terminate()
            try:
                # 给子进程 5秒时间优雅退出
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                LOGGER.warning("子进程未响应，强制杀死")
                process.kill()
            raise

    async def run_single_window(self):
        """运行单个交易窗口的采集"""
        now = datetime.now(TZ)
        window_end = get_current_window_end(now)

        if not window_end:
            LOGGER.warning("当前不在交易窗口内")
            return

        LOGGER.info(f"开始采集，窗口结束时间: {window_end.strftime('%H:%M:%S')}")

        # 启动采集任务
        self.collector_task = asyncio.create_task(self.start_collector())

        # 等待窗口结束
        try:
            await self.wait_until(window_end)
        except asyncio.CancelledError:
            pass

        # 停止采集
        if self.collector_task and not self.collector_task.done():
            self.collector_task.cancel()
            try:
                await self.collector_task
            except asyncio.CancelledError:
                pass

        LOGGER.info("窗口结束，采集暂停")

    async def run(self):
        """主循环"""
        LOGGER.info("=" * 60)
        LOGGER.info("  GetData V2 智能调度器启动")
        LOGGER.info(f"  时区: {TIMEZONE}")
        LOGGER.info("=" * 60)

        self.running = True

        while self.running:
            now = datetime.now(TZ)

            # Step 1: 找到下一个交易日
            if now.time() > TRADING_WINDOWS[-1][1]:
                # 今天的交易已结束，查明天
                check_date = now.date() + timedelta(days=1)
            else:
                check_date = now.date()

            next_trading_day = await self.find_next_trading_day(check_date)
            LOGGER.info(f"下一个交易日: {next_trading_day}")

            # Step 2: 计算下一个窗口开始时间
            if next_trading_day > now.date():
                # 交易日在未来
                first_window_start = datetime.combine(
                    next_trading_day, TRADING_WINDOWS[0][0], tzinfo=TZ
                )
                window_name = "T1"
            else:
                # 交易日是今天
                first_window_start, window_name = get_next_window_start(now)

            LOGGER.info(
                f"下一个窗口: {window_name} @ {first_window_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Step 3: 等待到窗口开始
            if first_window_start > now:
                await self.wait_until(first_window_start)

            # Step 4: 运行当天所有窗口
            now = datetime.now(TZ)
            while now.date() == next_trading_day and is_in_trading_window(now):
                await self.run_single_window()

                # 检查下一个窗口
                now = datetime.now(TZ)
                next_start, window_name = get_next_window_start(now)

                if next_start.date() == now.date() and next_start > now:
                    # 今天还有窗口
                    LOGGER.info(f"等待下一个窗口: {window_name}")
                    await self.wait_until(next_start)
                    now = datetime.now(TZ)
                else:
                    break

            LOGGER.info("今日采集完成，等待下一个交易日...")
            await asyncio.sleep(60)  # 短暂休息后重新检测

    def stop(self):
        """停止调度器"""
        self.running = False
        if self.collector_task:
            self.collector_task.cancel()


# ========= 入口 =========
async def main():
    scheduler = Scheduler()

    try:
        await scheduler.run()
    except KeyboardInterrupt:
        LOGGER.info("收到停止信号")
        scheduler.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已停止")
