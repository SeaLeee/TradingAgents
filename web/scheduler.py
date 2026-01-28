"""
定时任务调度器
处理AI模拟交易的每日信号检查和自动备份
"""

import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, List

# 简单的定时任务调度器（不依赖APScheduler）
# 如果安装了APScheduler，可以使用更高级的功能


class SimpleScheduler:
    """简单的定时任务调度器"""

    def __init__(self):
        self.jobs = {}
        self.running = False
        self.thread = None

    def add_job(
        self,
        func: Callable,
        job_id: str,
        interval_seconds: int = 3600,
        run_immediately: bool = False
    ):
        """添加定时任务"""
        self.jobs[job_id] = {
            "func": func,
            "interval": interval_seconds,
            "last_run": None,
            "next_run": datetime.utcnow() if run_immediately else datetime.utcnow() + timedelta(seconds=interval_seconds)
        }

    def remove_job(self, job_id: str):
        """移除定时任务"""
        if job_id in self.jobs:
            del self.jobs[job_id]

    def _run_loop(self):
        """运行循环"""
        import time
        while self.running:
            now = datetime.utcnow()
            for job_id, job in list(self.jobs.items()):
                if job["next_run"] <= now:
                    try:
                        job["func"]()
                    except Exception as e:
                        print(f"Scheduler job {job_id} failed: {e}")
                    job["last_run"] = now
                    job["next_run"] = now + timedelta(seconds=job["interval"])
            time.sleep(60)  # 每分钟检查一次

    def start(self):
        """启动调度器"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Scheduler started")

    def stop(self):
        """停止调度器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Scheduler stopped")


# 全局调度器实例
scheduler = SimpleScheduler()


def init_scheduler():
    """初始化调度器并添加默认任务"""
    # 每日模拟交易检查（每小时检查一次，在交易时间执行）
    scheduler.add_job(
        func=daily_simulation_check,
        job_id="daily_simulation_check",
        interval_seconds=3600,  # 1小时
        run_immediately=False
    )

    # 每周自动备份（每7天）
    scheduler.add_job(
        func=weekly_auto_backup,
        job_id="weekly_auto_backup",
        interval_seconds=7 * 24 * 3600,  # 7天
        run_immediately=False
    )

    scheduler.start()


def daily_simulation_check():
    """每日模拟交易检查"""
    from .ai_simulation_service import check_all_active_simulations

    print(f"[{datetime.utcnow()}] Running daily simulation check...")

    # 检查是否在交易时间（美东时间9:30-16:00）
    # 简化处理：UTC时间14:30-21:00
    now = datetime.utcnow()
    if now.hour < 14 or now.hour > 21:
        print("Outside trading hours, skipping...")
        return

    try:
        results = check_all_active_simulations()
        print(f"Simulation check completed: {len(results)} sessions processed")
        for result in results:
            if "error" in result:
                print(f"  Session {result['session_id']}: Error - {result['error']}")
            else:
                status = result.get("result", {}).get("status", "unknown")
                print(f"  Session {result['session_id']} ({result['ticker']}): {status}")
    except Exception as e:
        print(f"Simulation check failed: {e}")


def weekly_auto_backup():
    """每周自动备份"""
    from .backup_service import create_backup

    print(f"[{datetime.utcnow()}] Running weekly auto backup...")

    try:
        result = create_backup(
            backup_type="full",
            format="json",
            destination="local",
            user_id=None  # 系统备份
        )

        if result.get("success"):
            print(f"Auto backup completed: {result.get('file_path')}")
        else:
            print(f"Auto backup failed: {result.get('error')}")
    except Exception as e:
        print(f"Auto backup failed: {e}")


def trigger_simulation_check():
    """手动触发模拟交易检查"""
    daily_simulation_check()


def trigger_backup():
    """手动触发备份"""
    weekly_auto_backup()


def get_scheduler_status() -> Dict:
    """获取调度器状态"""
    jobs_status = []
    for job_id, job in scheduler.jobs.items():
        jobs_status.append({
            "job_id": job_id,
            "interval_seconds": job["interval"],
            "last_run": job["last_run"].isoformat() if job["last_run"] else None,
            "next_run": job["next_run"].isoformat() if job["next_run"] else None
        })

    return {
        "running": scheduler.running,
        "jobs": jobs_status
    }


# 尝试使用APScheduler（如果已安装）
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    class APSchedulerWrapper:
        """APScheduler包装器"""

        def __init__(self):
            self.scheduler = BackgroundScheduler()

        def add_cron_job(
            self,
            func: Callable,
            job_id: str,
            hour: int = 9,
            minute: int = 35,
            day_of_week: str = None
        ):
            """添加Cron定时任务"""
            trigger_kwargs = {"hour": hour, "minute": minute}
            if day_of_week:
                trigger_kwargs["day_of_week"] = day_of_week

            self.scheduler.add_job(
                func,
                CronTrigger(**trigger_kwargs),
                id=job_id,
                replace_existing=True
            )

        def add_interval_job(
            self,
            func: Callable,
            job_id: str,
            hours: int = 0,
            minutes: int = 0,
            seconds: int = 0
        ):
            """添加间隔定时任务"""
            self.scheduler.add_job(
                func,
                IntervalTrigger(hours=hours, minutes=minutes, seconds=seconds),
                id=job_id,
                replace_existing=True
            )

        def remove_job(self, job_id: str):
            """移除任务"""
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass

        def start(self):
            """启动"""
            self.scheduler.start()

        def stop(self):
            """停止"""
            self.scheduler.shutdown()

        def get_jobs(self):
            """获取所有任务"""
            return self.scheduler.get_jobs()

    # 如果APScheduler可用，替换简单调度器
    def init_apscheduler():
        """使用APScheduler初始化"""
        ap_scheduler = APSchedulerWrapper()

        # 每日模拟交易检查（美股开盘后5分钟）
        ap_scheduler.add_cron_job(
            func=daily_simulation_check,
            job_id="daily_simulation_check",
            hour=14,  # UTC 14:35 = EST 9:35
            minute=35
        )

        # 每周自动备份（周日凌晨2点）
        ap_scheduler.add_cron_job(
            func=weekly_auto_backup,
            job_id="weekly_auto_backup",
            hour=2,
            minute=0,
            day_of_week="sun"
        )

        ap_scheduler.start()
        return ap_scheduler

    APSCHEDULER_AVAILABLE = True

except ImportError:
    APSCHEDULER_AVAILABLE = False


def start_scheduler(use_apscheduler: bool = True):
    """启动调度器"""
    if use_apscheduler and APSCHEDULER_AVAILABLE:
        print("Starting APScheduler...")
        return init_apscheduler()
    else:
        print("Starting simple scheduler...")
        init_scheduler()
        return scheduler
