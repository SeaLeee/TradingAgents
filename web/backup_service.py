"""
数据库备份服务
支持导出到JSON/SQLite，备份到GitHub、阿里云盘、本地
"""

import os
import json
import shutil
import base64
from datetime import datetime
from typing import Dict, List, Optional
import requests

from .database import (
    get_db, engine, Base, BackupRecord,
    User, Session, AnalysisHistory, TradingJournal,
    SimPortfolio, SimTrade, SimPosition, SimDailySnapshot,
    Strategy, BacktestResult, BatchBacktestJob, AISimulationSession,
    StockStrategyMatch, StockPersonality,
    create_backup_record, update_backup_record, get_backup_records,
    DATABASE_URL
)


# 备份目录
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "data", "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)


# 所有可备份的表模型
TABLE_MODELS = {
    "users": User,
    "sessions": Session,
    "analysis_history": AnalysisHistory,
    "trading_journals": TradingJournal,
    "sim_portfolios": SimPortfolio,
    "sim_trades": SimTrade,
    "sim_positions": SimPosition,
    "sim_daily_snapshots": SimDailySnapshot,
    "strategies": Strategy,
    "backtest_results": BacktestResult,
    "batch_backtest_jobs": BatchBacktestJob,
    "ai_simulation_sessions": AISimulationSession,
    "stock_strategy_matches": StockStrategyMatch,
    "stock_personalities": StockPersonality,
    "backup_records": BackupRecord,
}


def create_backup(
    backup_type: str = "full",
    format: str = "json",
    destination: str = "local",
    user_id: int = None,
    custom_path: str = None,
    github_config: Dict = None,
    aliyun_config: Dict = None
) -> Dict:
    """
    创建数据库备份

    参数:
    - backup_type: full, strategies_only, user_data, trades_only
    - format: json, sqlite
    - destination: local, github, aliyun_drive
    - user_id: 用户ID（可选）
    - custom_path: 自定义保存路径
    - github_config: {"token": "xxx", "repo": "user/repo", "path": "backups/"}
    - aliyun_config: {"refresh_token": "xxx", "folder": "/backups/"}
    """
    with get_db() as db:
        # 创建备份记录
        record = create_backup_record(
            db=db,
            backup_type=backup_type,
            format=format,
            destination=destination,
            user_id=user_id,
            file_path=custom_path,
            github_repo=github_config.get("repo") if github_config else None,
            github_path=github_config.get("path") if github_config else None,
            aliyun_folder=aliyun_config.get("folder") if aliyun_config else None
        )

        try:
            # 更新状态为运行中
            update_backup_record(db, record, status="running")

            # 确定要备份的表
            tables = get_tables_for_backup_type(backup_type)

            # 导出数据
            if format == "json":
                backup_data = export_to_json(tables, user_id)
                file_ext = ".json"
            else:  # sqlite
                backup_data = None
                file_ext = ".db"

            # 生成文件名
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"tradingagents_backup_{backup_type}_{timestamp}{file_ext}"

            # 根据目标位置保存
            if destination == "local":
                file_path = save_to_local(backup_data, filename, custom_path, format)
                result = {"file_path": file_path}

            elif destination == "github":
                if not github_config or not github_config.get("token"):
                    raise ValueError("GitHub配置缺失")
                result = backup_to_github(
                    backup_data=backup_data,
                    filename=filename,
                    repo=github_config["repo"],
                    path=github_config.get("path", "backups/"),
                    token=github_config["token"]
                )
                file_path = result.get("html_url", github_config.get("path", "") + filename)

            elif destination == "aliyun_drive":
                if not aliyun_config or not aliyun_config.get("refresh_token"):
                    raise ValueError("阿里云盘配置缺失")
                result = backup_to_aliyun_drive(
                    backup_data=backup_data,
                    filename=filename,
                    folder_path=aliyun_config.get("folder", "/TradingAgents/backups/"),
                    refresh_token=aliyun_config["refresh_token"]
                )
                file_path = result.get("file_path", aliyun_config.get("folder", "") + filename)

            else:
                raise ValueError(f"不支持的目标位置: {destination}")

            # 计算文件大小
            file_size = len(json.dumps(backup_data)) if backup_data else 0
            file_size_str = format_file_size(file_size)

            # 记录数量统计
            record_counts = {}
            if backup_data and "data" in backup_data:
                for table, records in backup_data["data"].items():
                    record_counts[table] = len(records)

            # 更新备份记录
            update_backup_record(
                db, record,
                status="completed",
                file_path=file_path,
                tables_included=json.dumps(tables),
                record_counts=json.dumps(record_counts),
                file_size=file_size_str,
                completed_at=datetime.utcnow()
            )

            return {
                "success": True,
                "backup_id": record.id,
                "file_path": file_path,
                "file_size": file_size_str,
                "tables": tables,
                "record_counts": record_counts,
                "result": result
            }

        except Exception as e:
            update_backup_record(
                db, record,
                status="failed",
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            return {"error": str(e), "backup_id": record.id}


def get_tables_for_backup_type(backup_type: str) -> List[str]:
    """根据备份类型获取要备份的表"""
    if backup_type == "full":
        return list(TABLE_MODELS.keys())
    elif backup_type == "strategies_only":
        return ["strategies", "backtest_results", "batch_backtest_jobs"]
    elif backup_type == "user_data":
        return ["users", "analysis_history", "trading_journals"]
    elif backup_type == "trades_only":
        return ["sim_portfolios", "sim_trades", "sim_positions", "sim_daily_snapshots"]
    elif backup_type == "stock_library":
        return ["stock_strategy_matches", "stock_personalities", "ai_simulation_sessions"]
    else:
        return list(TABLE_MODELS.keys())


def export_to_json(tables: List[str] = None, user_id: int = None) -> Dict:
    """
    导出数据库到JSON

    返回格式:
    {
        "metadata": {
            "version": "1.0",
            "created_at": "2025-01-28T10:00:00",
            "tables": ["users", "strategies", ...]
        },
        "data": {
            "users": [...],
            "strategies": [...],
            ...
        }
    }
    """
    with get_db() as db:
        result = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "tables": [],
                "export_type": "full" if not user_id else "user_specific"
            },
            "data": {}
        }

        tables_to_export = tables or list(TABLE_MODELS.keys())

        for table_name in tables_to_export:
            if table_name not in TABLE_MODELS:
                continue

            model = TABLE_MODELS[table_name]

            try:
                query = db.query(model)

                # 如果指定了用户ID，只导出该用户的数据
                if user_id and hasattr(model, 'user_id'):
                    query = query.filter(model.user_id == user_id)

                records = query.all()

                # 转换为字典
                result["data"][table_name] = []
                for record in records:
                    if hasattr(record, 'to_dict'):
                        result["data"][table_name].append(record.to_dict())
                    else:
                        # 基本转换
                        record_dict = {}
                        for column in record.__table__.columns:
                            value = getattr(record, column.name)
                            if isinstance(value, datetime):
                                value = value.isoformat()
                            record_dict[column.name] = value
                        result["data"][table_name].append(record_dict)

                result["metadata"]["tables"].append(table_name)

            except Exception as e:
                result["data"][table_name] = {"error": str(e)}

        return result


def export_to_sqlite(output_path: str = None) -> str:
    """
    导出数据库为SQLite文件

    对于已经是SQLite的情况，直接复制
    """
    if output_path is None:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(BACKUP_DIR, f"tradingagents_backup_{timestamp}.db")

    if "sqlite" in DATABASE_URL:
        # 直接复制SQLite文件
        db_path = DATABASE_URL.replace("sqlite:///", "")
        shutil.copy(db_path, output_path)
    else:
        # 从PostgreSQL导出到SQLite需要更复杂的迁移逻辑
        raise NotImplementedError("PostgreSQL到SQLite的导出暂未实现")

    return output_path


def save_to_local(backup_data: Dict, filename: str, custom_path: str = None, format: str = "json") -> str:
    """保存备份到本地"""
    if custom_path:
        os.makedirs(custom_path, exist_ok=True)
        file_path = os.path.join(custom_path, filename)
    else:
        file_path = os.path.join(BACKUP_DIR, filename)

    if format == "json":
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
    else:  # sqlite
        file_path = export_to_sqlite(file_path)

    return file_path


def backup_to_github(
    backup_data: Dict,
    filename: str,
    repo: str,
    path: str,
    token: str,
    message: str = None
) -> Dict:
    """
    备份到GitHub

    使用GitHub API创建或更新文件
    """
    content = json.dumps(backup_data, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(content.encode()).decode()

    file_path = path.rstrip('/') + '/' + filename
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 检查文件是否存在（获取SHA）
    response = requests.get(url, headers=headers)
    sha = response.json().get("sha") if response.status_code == 200 else None

    # 创建或更新文件
    data = {
        "message": message or f"Backup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": encoded,
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    response = requests.put(url, json=data, headers=headers)

    if response.status_code in [200, 201]:
        result = response.json()
        return {
            "success": True,
            "html_url": result.get("content", {}).get("html_url"),
            "sha": result.get("content", {}).get("sha"),
            "file_path": file_path
        }
    else:
        return {
            "success": False,
            "error": response.json().get("message", "Unknown error"),
            "status_code": response.status_code
        }


def backup_to_aliyun_drive(
    backup_data: Dict,
    filename: str,
    folder_path: str,
    refresh_token: str
) -> Dict:
    """
    备份到阿里云盘

    使用阿里云盘API上传文件
    注意：阿里云盘API需要先通过refresh_token获取access_token
    """
    try:
        # 1. 获取 access_token
        auth_url = "https://openapi.alipan.com/oauth/access_token"
        auth_response = requests.post(auth_url, json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        })

        if auth_response.status_code != 200:
            return {
                "success": False,
                "error": "无法获取阿里云盘访问令牌",
                "details": auth_response.text
            }

        auth_data = auth_response.json()
        access_token = auth_data.get("access_token")
        drive_id = auth_data.get("default_drive_id")

        if not access_token or not drive_id:
            return {
                "success": False,
                "error": "阿里云盘认证失败"
            }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # 2. 获取或创建文件夹
        folder_id = get_or_create_aliyun_folder(drive_id, folder_path, headers)

        if not folder_id:
            return {
                "success": False,
                "error": "无法创建阿里云盘文件夹"
            }

        # 3. 创建文件
        content = json.dumps(backup_data, ensure_ascii=False, indent=2)
        content_bytes = content.encode('utf-8')

        # 创建文件请求
        create_url = "https://openapi.alipan.com/adrive/v1.0/openFile/create"
        create_response = requests.post(create_url, headers=headers, json={
            "drive_id": drive_id,
            "parent_file_id": folder_id,
            "name": filename,
            "type": "file",
            "check_name_mode": "auto_rename",
            "size": len(content_bytes)
        })

        if create_response.status_code != 200:
            return {
                "success": False,
                "error": "创建文件失败",
                "details": create_response.text
            }

        create_data = create_response.json()
        upload_url = create_data.get("part_info_list", [{}])[0].get("upload_url")
        file_id = create_data.get("file_id")

        if not upload_url:
            return {
                "success": False,
                "error": "获取上传URL失败"
            }

        # 4. 上传文件内容
        upload_response = requests.put(upload_url, data=content_bytes)

        if upload_response.status_code not in [200, 201]:
            return {
                "success": False,
                "error": "上传文件内容失败"
            }

        # 5. 完成上传
        complete_url = "https://openapi.alipan.com/adrive/v1.0/openFile/complete"
        complete_response = requests.post(complete_url, headers=headers, json={
            "drive_id": drive_id,
            "file_id": file_id,
            "upload_id": create_data.get("upload_id")
        })

        if complete_response.status_code == 200:
            return {
                "success": True,
                "file_id": file_id,
                "file_path": folder_path + filename
            }
        else:
            return {
                "success": False,
                "error": "完成上传失败",
                "details": complete_response.text
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_or_create_aliyun_folder(drive_id: str, folder_path: str, headers: dict) -> Optional[str]:
    """获取或创建阿里云盘文件夹"""
    # 简化实现：使用根目录
    # 完整实现需要逐级创建文件夹
    return "root"


def restore_from_json(json_data: Dict, restore_type: str = "full", user_id: int = None) -> Dict:
    """从JSON恢复数据库"""
    if "data" not in json_data:
        return {"error": "无效的备份数据格式"}

    with get_db() as db:
        restored_counts = {}

        for table_name, records in json_data["data"].items():
            if table_name not in TABLE_MODELS:
                continue

            if isinstance(records, dict) and "error" in records:
                continue

            model = TABLE_MODELS[table_name]
            count = 0

            for record_data in records:
                try:
                    # 检查是否已存在
                    if 'id' in record_data:
                        existing = db.query(model).filter(model.id == record_data['id']).first()
                        if existing:
                            continue

                    # 创建新记录
                    new_record = model(**record_data)
                    db.add(new_record)
                    count += 1
                except Exception as e:
                    continue

            restored_counts[table_name] = count

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            return {"error": f"恢复失败: {str(e)}"}

        return {
            "success": True,
            "restored_counts": restored_counts
        }


def restore_from_sqlite(sqlite_path: str) -> Dict:
    """从SQLite文件恢复"""
    if not os.path.exists(sqlite_path):
        return {"error": "备份文件不存在"}

    # 对于SQLite到SQLite的恢复，可以直接替换
    if "sqlite" in DATABASE_URL:
        db_path = DATABASE_URL.replace("sqlite:///", "")

        # 备份当前数据库
        backup_current = db_path + ".bak"
        shutil.copy(db_path, backup_current)

        try:
            shutil.copy(sqlite_path, db_path)
            return {
                "success": True,
                "message": "数据库已恢复",
                "previous_backup": backup_current
            }
        except Exception as e:
            # 恢复失败，还原备份
            shutil.copy(backup_current, db_path)
            return {"error": f"恢复失败: {str(e)}"}
    else:
        return {"error": "跨数据库类型恢复暂未实现"}


def get_backup_history(user_id: int = None, limit: int = 50) -> List[Dict]:
    """获取备份历史"""
    with get_db() as db:
        records = get_backup_records(db, user_id, limit)
        return [r.to_dict() for r in records]


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def download_backup(backup_id: int) -> Dict:
    """下载备份文件"""
    with get_db() as db:
        record = db.query(BackupRecord).filter(BackupRecord.id == backup_id).first()
        if not record:
            return {"error": "备份记录不存在"}

        if record.destination == "local" and record.file_path:
            if os.path.exists(record.file_path):
                return {
                    "success": True,
                    "file_path": record.file_path,
                    "filename": os.path.basename(record.file_path)
                }
            else:
                return {"error": "备份文件不存在"}
        else:
            return {"error": "该备份不支持直接下载"}
