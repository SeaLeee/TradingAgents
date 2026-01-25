"""
部署后验证脚本
用于检查新功能是否正常部署
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.database import get_db, get_strategies, Strategy

def check_deployment():
    """检查部署状态"""
    print("🔍 检查部署状态...")
    print("-" * 50)
    
    try:
        with get_db() as db:
            # 检查策略表是否存在
            strategies = get_strategies(db, public_only=True, active_only=True)
            
            if strategies:
                print(f"✅ 策略表已创建，找到 {len(strategies)} 个策略")
                print("\n策略列表：")
                for i, strategy in enumerate(strategies[:5], 1):  # 只显示前5个
                    print(f"  {i}. {strategy.name} ({strategy.strategy_type})")
                if len(strategies) > 5:
                    print(f"  ... 还有 {len(strategies) - 5} 个策略")
            else:
                print("⚠️  策略表存在但未找到策略，可能需要初始化")
            
            # 检查表结构
            from sqlalchemy import inspect
            inspector = inspect(db.bind)
            tables = inspector.get_table_names()
            
            required_tables = ['strategies', 'backtest_results']
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                print(f"\n❌ 缺少表: {', '.join(missing_tables)}")
                print("   请检查数据库初始化是否成功")
            else:
                print(f"\n✅ 所有必需的表已创建: {', '.join(required_tables)}")
            
            print("\n" + "-" * 50)
            print("✅ 部署检查完成！")
            
    except Exception as e:
        print(f"\n❌ 检查失败: {e}")
        print("\n可能的原因：")
        print("1. 数据库连接失败")
        print("2. 表尚未创建")
        print("3. 需要重新部署")
        return False
    
    return True

if __name__ == "__main__":
    check_deployment()
