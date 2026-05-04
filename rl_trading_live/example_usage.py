"""
使用示例
Quick Start Guide
"""

def example_basic():
    """基础使用：获取单个交易信号"""
    print("=== 示例 1: 获取单个交易信号 ===")
    
    from rl_trading_live import get_trading_signal
    
    position_info = {
        "position_count": 0,      # 当前持仓0颗
        "floating_pnl_pct": 0,    # 无浮盈浮亏
        "hold_days": 0,            # 持仓0天
        "avg_cost_price": 0,       # 无成本
        "entry_max_drawdown": 0,   # 无回撤
        "profit_run_days": 0       # 无连续盈利
    }
    
    signal = get_trading_signal("002156", position_info)
    
    print(f"股票: {signal['stock']['code']}")
    print(f"价格: {signal['stock']['current_price']:.2f}")
    print(f"动作: {signal['trading_signal']['action']} - {signal['trading_signal']['description']}")
    print(f"风控建议: 止损价={signal['risk_control']['hard_stop_loss_price']:.2f}")
    print(f"备注: {signal['execution_suggestion']['notes']}\n")

def example_train_first():
    """示例2：先训练模型再使用"""
    print("=== 示例 2: 训练 + 获取信号 ===")
    
    from rl_trading_live import train_model, get_trading_signal
    
    # 第一步：训练模型 (只需要执行一次)
    print("1. 训练模型...")
    model_path = train_model("002079")
    print(f"   模型已保存: {model_path}\n")
    
    # 第二步：获取信号
    print("2. 获取交易信号...")
    signal = get_trading_signal("002079")
    print(f"   建议动作: {signal['trading_signal']['description']}\n")

def example_risk_control():
    """示例3：风控检查"""
    print("=== 示例 3: 风控检查 ===")
    
    from rl_trading_live import check_stop_loss, apply_risk_controls
    
    position = {
        "position_count": 5,
        "floating_pnl_pct": -6,  # 浮亏6%，触发硬止损
        "hold_days": 10,
        "entry_max_drawdown": 2
    }
    
    # 检查止损
    stop_loss_result = check_stop_loss(position)
    print(f"风控触发: {stop_loss_result['should_stop_loss']}")
    print(f"建议动作: {stop_loss_result['action']}")
    print(f"原因: {stop_loss_result['reason']}")
    print(f"风险级别: {stop_loss_result['level']}\n")

def example_openclaw_entry():
    """示例4：OpenClaw标准入口调用"""
    print("=== 示例 4: OpenClaw 入口调用 ===")
    
    from rl_trading_live.openclaw_entry import execute
    
    # 健康检查
    print("1. 健康检查:")
    health_result = execute({"action": "health"})
    print(f"   状态: {health_result['status']}")
    print(f"   版本: {health_result['skill_version']}")
    
    # 获取信号
    print("\n2. 获取交易信号:")
    params = {
        "action": "signal",
        "stock_code": "002156",
        "position_info": {"position_count": 0},
        "dry_run": True
    }
    result = execute(params)
    print(f"   状态: {result['status']}")
    print(f"   动作: {result['data']['trading_signal']['description']}\n")

if __name__ == "__main__":
    # 运行所有示例
    try:
        example_basic()
    except Exception as e:
        print(f"示例1异常 (可能需要先训练模型): {e}\n")
    
    # example_train_first()  # 如需训练取消注释
    
    example_risk_control()
    example_openclaw_entry()
    
    print("=== 示例完成 ===")
    print("\n提示：")
    print("- 首次使用请先执行 train_model() 训练模型")
    print("- 实盘交易请将 dry_run 设为 False")
    print("- 严格遵循风控模块建议")
