"""
OpenClaw 标准入口
Skill ID: rl-trading-live
Version: 2.0.0
"""
from typing import Dict, Any
from .signal_generator import get_trading_signal, batch_get_signals, train_model
from .trade_executor import execute_trade, BaseTradeAPI
from .config import CONFIG

def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenClaw 标准执行入口
    
    Args:
        params: {
            "action": "train|signal|batch|execute",  # 执行动作类型
            "stock_code": "002156",                   # 单个股票代码
            "stock_list": [{"code": "...", "position": {...}}],  # 批量
            "position_info": {...},                   # 仓位信息
            "dry_run": True,                          # 是否模拟
            "trade_api": None                         # 交易API实例
        }
    
    Returns:
        标准响应
    """
    action = params.get("action", "signal")
    dry_run = params.get("dry_run", True)
    
    try:
        # ========== 1. 训练模型 ==========
        if action == "train":
            stock_code = params.get("stock_code")
            if not stock_code:
                return _error_response("缺少 stock_code 参数")
            
            model_path = train_model(stock_code)
            return {
                "skill_id": "rl-trading-live",
                "skill_version": CONFIG['version'],
                "action": "train",
                "status": "success",
                "stock_code": stock_code,
                "model_path": model_path
            }
        
        # ========== 2. 获取单个信号 ==========
        elif action == "signal":
            stock_code = params.get("stock_code")
            if not stock_code:
                return _error_response("缺少 stock_code 参数")
            
            position_info = params.get("position_info", {})
            signal = get_trading_signal(stock_code, position_info)
            
            return {
                "skill_id": "rl-trading-live",
                "skill_version": CONFIG['version'],
                "action": "signal",
                "status": "success",
                "data": signal
            }
        
        # ========== 3. 批量获取信号 ==========
        elif action == "batch":
            stock_list = params.get("stock_list", [])
            if not stock_list:
                return _error_response("缺少 stock_list 参数")
            
            result = batch_get_signals(stock_list)
            result["skill_id"] = "rl-trading-live"
            result["skill_version"] = CONFIG['version']
            return result
        
        # ========== 4. 执行交易 ==========
        elif action == "execute":
            signal = params.get("signal")
            trade_api = params.get("trade_api")
            
            if not signal:
                return _error_response("缺少 signal 参数")
            
            if not trade_api or not isinstance(trade_api, BaseTradeAPI):
                return _error_response("需要有效的 trade_api 实例 (继承 BaseTradeAPI)")
            
            exec_result = execute_trade(signal, trade_api, dry_run=dry_run)
            
            return {
                "skill_id": "rl-trading-live",
                "skill_version": CONFIG['version'],
                "action": "execute",
                "status": "success",
                "dry_run": dry_run,
                "data": exec_result
            }
        
        # ========== 5. 健康检查 ==========
        elif action == "health":
            return {
                "skill_id": "rl-trading-live",
                "skill_version": CONFIG['version'],
                "status": "healthy",
                "config": {
                    "bullet_size_pct": CONFIG['bullet_size_pct'],
                    "max_bullets": CONFIG['max_bullets_per_stock'],
                    "hard_stop_loss": CONFIG['risk_control']['hard_stop_loss_pct']
                }
            }
        
        else:
            return _error_response(f"未知 action: {action}, 可选: train, signal, batch, execute, health")
    
    except Exception as e:
        return _error_response(f"执行异常: {str(e)}", exception=str(e))

def _error_response(message: str, **kwargs) -> Dict[str, Any]:
    """标准错误响应"""
    return {
        "skill_id": "rl-trading-live",
        "skill_version": CONFIG['version'],
        "status": "error",
        "message": message,
        **kwargs
    }

# 兼容旧版本入口
main = execute
