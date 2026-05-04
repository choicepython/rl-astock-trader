"""
风险控制模块
"""
from .config import CONFIG

def apply_risk_controls(signal: dict, position_info: dict) -> dict:
    """
    应用风控规则，可能修改原始信号
    
    Args:
        signal: 原始交易信号
        position_info: 仓位信息
        
    Returns:
        修改后的信号 + 风控标记
    """
    risk_flags = []
    modified_signal = signal.copy()
    action = signal['trading_signal']['action']
    current_bullets = position_info.get('position_count', 0)
    floating_pnl = position_info.get('floating_pnl_pct', 0)
    
    # ========== 强制风控规则 ==========
    
    # 1. 硬止损检查
    if floating_pnl <= CONFIG['risk_control']['hard_stop_loss_pct']:
        risk_flags.append(f"HARD_STOP_LOSS_TRIGGERED: 浮亏{floating_pnl:.1f}%达到硬止损线")
        modified_signal['trading_signal']['action'] = 'close_all'
        modified_signal['trading_signal']['description'] = '【风控强制】全部清仓'
        modified_signal['trading_signal']['urgency'] = 'critical'
    
    # 2. 最大回撤检查
    if position_info.get('entry_max_drawdown', 0) >= CONFIG['risk_control']['max_drawdown_reduce_pct']:
        if current_bullets > 3:
            risk_flags.append(f"DRAWDOWN_REDUCE: 回撤达到{position_info['entry_max_drawdown']:.1f}%，建议减仓至3颗")
            if 'add' in action:
                modified_signal['trading_signal']['action'] = 'reduce_3'
                modified_signal['trading_signal']['description'] = '【风控减仓】减仓3颗'
    
    # 3. 持仓超时检查
    if position_info.get('hold_days', 0) >= CONFIG['risk_control']['max_hold_days']:
        risk_flags.append(f"HOLD_TIMEOUT: 持仓{position_info['hold_days']}天超时")
        modified_signal['trading_signal']['action'] = 'reduce_3'
        modified_signal['trading_signal']['description'] = '【风控超时】分批减仓'
    
    # 4. 仓位上限检查
    if 'add' in action and current_bullets >= CONFIG['risk_control']['max_position_bullets']:
        risk_flags.append(f"POSITION_LIMIT: 已达最大仓位{CONFIG['risk_control']['max_position_bullets']}颗")
        modified_signal['trading_signal']['action'] = 'hold'
        modified_signal['trading_signal']['description'] = '【风控限制】已达上限，改为观望'
    
    # 5. 逆势加仓惩罚 - 浮亏状态下禁止加仓超过1颗
    if 'add' in action and floating_pnl < 0:
        if action == 'add_2' or action == 'add_3':
            risk_flags.append(f"RISING_AGAINST_TREND: 浮亏状态下禁止大额加仓")
            modified_signal['trading_signal']['action'] = 'add_1'
            modified_signal['trading_signal']['description'] = '【风控调整】浮亏时仅允许加1颗'
    
    # 6. 满仓禁止任何加仓
    if 'add' in action and current_bullets >= 10:
        risk_flags.append("FULL_POSITION: 已满仓，禁止加仓")
        modified_signal['trading_signal']['action'] = 'hold'
        modified_signal['trading_signal']['description'] = '【风控满仓】已满仓，观望'
    
    modified_signal['risk_controls_applied'] = risk_flags
    modified_signal['risk_level'] = 'low' if len(risk_flags) == 0 else 'high'
    
    return modified_signal

def check_stop_loss(position_info: dict) -> dict:
    """
    仅检查是否需要止损 (独立函数，可用于轮询)
    
    Returns:
        {
            "should_stop_loss": bool,
            "action": "reduce_3|close_all",
            "reason": str
        }
    """
    floating_pnl = position_info.get('floating_pnl_pct', 0)
    drawdown = position_info.get('entry_max_drawdown', 0)
    hold_days = position_info.get('hold_days', 0)
    bullets = position_info.get('position_count', 0)
    
    # 硬止损 - 必须全清
    if floating_pnl <= CONFIG['risk_control']['hard_stop_loss_pct']:
        return {
            "should_stop_loss": True,
            "action": "close_all",
            "reason": f"硬止损触发: 浮亏{floating_pnl:.1f}%",
            "level": "critical"
        }
    
    # 大回撤 - 减仓
    if drawdown >= CONFIG['risk_control']['max_drawdown_reduce_pct']:
        return {
            "should_stop_loss": True,
            "action": "reduce_3" if bullets > 3 else "close_all",
            "reason": f"回撤过大: {drawdown:.1f}%",
            "level": "warning"
        }
    
    # 超时 - 减仓
    if hold_days >= CONFIG['risk_control']['max_hold_days']:
        return {
            "should_stop_loss": True,
            "action": "reduce_3",
            "reason": f"持仓超时: {hold_days}天",
            "level": "warning"
        }
    
    return {
        "should_stop_loss": False,
        "action": "hold",
        "reason": "无风控触发",
        "level": "normal"
    }

def get_risk_summary(positions: list) -> dict:
    """
    获取整体组合风险概要
    
    Args:
        positions: 所有持仓列表
    
    Returns:
        风险概要
    """
    total_bullets = sum(p.get('position_count', 0) for p in positions)
    max_single_position = max(p.get('position_count', 0) for p in positions) if positions else 0
    total_exposure = total_bullets * CONFIG['bullet_size_pct'] / len(positions) if positions else 0
    
    under_stop_loss = sum(1 for p in positions 
                        if p.get('floating_pnl_pct', 0) <= CONFIG['risk_control']['hard_stop_loss_pct'])
    
    high_drawdown = sum(1 for p in positions
                       if p.get('entry_max_drawdown', 0) >= CONFIG['risk_control']['max_drawdown_reduce_pct'])
    
    risk_level = "low"
    if under_stop_loss > 0 or high_drawdown > 0 or total_bullets >= 8 * len(positions):
        risk_level = "high"
    elif total_bullets >= 5 * len(positions) or max_single_position >= 7:
        risk_level = "medium"
    
    return {
        "total_positions": len(positions),
        "total_bullets_used": total_bullets,
        "exposure_pct": total_exposure,
        "max_single_position": max_single_position,
        "at_stop_loss_count": under_stop_loss,
        "high_drawdown_count": high_drawdown,
        "overall_risk_level": risk_level,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
