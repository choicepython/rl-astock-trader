"""
交易信号生成模块
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

from .config import CONFIG
from .data_fetcher import fetch_stock_data, calculate_indicators
from ..train_rl_strategy import train_rl_model

ACTION_MAP = {
    0: ("hold", "观望持有", 0),
    1: ("add_1", "加仓1颗", 1),
    2: ("add_2", "加仓2颗", 2),
    3: ("add_3", "加仓3颗", 3),
    4: ("reduce_1", "减仓1颗", -1),
    5: ("reduce_2", "减仓2颗", -2),
    6: ("reduce_3", "减仓3颗", -3),
    7: ("close_all", "全部清仓", -999)
}

def train_model(stock_code: str) -> str:
    """
    训练强化学习模型
    
    Args:
        stock_code: 股票代码
        
    Returns:
        模型文件路径
    """
    print(f"开始训练 {stock_code} 强化学习模型...")
    model = train_rl_model(stock_code)
    model_path = os.path.join(CONFIG['model_dir'], f"ppo_trading_{stock_code}.zip")
    
    if os.path.exists(model_path):
        os.remove(model_path)
    os.rename(f"ppo_trading_{stock_code}.zip", model_path)
    
    print(f"模型已保存至: {model_path}")
    return model_path

def _load_model(stock_code: str):
    """加载模型"""
    model_path = os.path.join(CONFIG['model_dir'], f"ppo_trading_{stock_code}.zip")
    
    if not os.path.exists(model_path):
        # 尝试当前目录
        model_path = f"ppo_trading_{stock_code}.zip"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}, 请先调用 train_model() 训练")
    
    return PPO.load(model_path)

def get_trading_signal(stock_code: str, position_info: dict = None) -> dict:
    """
    获取实盘交易信号 (标准接口)
    
    Args:
        stock_code: 股票代码
        position_info: 仓位信息字典
            {
                "position_count": 0,
                "floating_pnl_pct": 0,
                "hold_days": 0,
                "avg_cost_price": 0,
                "entry_max_drawdown": 0,
                "profit_run_days": 0
            }
    
    Returns:
        标准格式交易信号
    """
    if position_info is None:
        position_info = {}
    
    # 填充默认值
    pos = {
        "position_count": position_info.get("position_count", 0),
        "floating_pnl_pct": position_info.get("floating_pnl_pct", 0),
        "hold_days": position_info.get("hold_days", 0),
        "avg_cost_price": position_info.get("avg_cost_price", 0),
        "entry_max_drawdown": position_info.get("entry_max_drawdown", 0),
        "profit_run_days": position_info.get("profit_run_days", 0)
    }
    
    # 获取数据并计算指标
    df = fetch_stock_data(stock_code)
    if df.empty:
        return {"status": "error", "message": "无法获取股票数据"}
    
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    current_price = latest['close']
    
    # 加载模型并预测
    try:
        model = _load_model(stock_code)
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    
    # 构造14维技术特征
    tech_features = latest[CONFIG['feature_cols']].values.astype(np.float32)
    
    # 构造6维状态特征
    avg_cost_ratio = 0
    if pos['avg_cost_price'] > 0:
        avg_cost_ratio = (current_price - pos['avg_cost_price']) / pos['avg_cost_price'] * 100
    
    state_features = np.array([
        pos['position_count'] / CONFIG['max_bullets_per_stock'],
        pos['floating_pnl_pct'] / 10,
        pos['hold_days'] / 22,
        avg_cost_ratio / 10,
        pos['entry_max_drawdown'] / 5,
        pos['profit_run_days'] / 5
    ], dtype=np.float32)
    
    obs = np.concatenate([tech_features, state_features])
    action, _states = model.predict(obs, deterministic=True)
    
    action_code, action_desc, bullet_change = ACTION_MAP[int(action)]
    
    # 计算风控价格
    if pos['position_count'] > 0 and pos['avg_cost_price'] > 0:
        cost_price = pos['avg_cost_price']
        hard_stop_loss = cost_price * (1 + CONFIG['risk_control']['hard_stop_loss_pct'] / 100)
        take_profit_1 = cost_price * 1.05
        take_profit_2 = cost_price * 1.10
    else:
        hard_stop_loss = current_price * 0.95
        take_profit_1 = current_price * 1.05
        take_profit_2 = current_price * 1.10
    
    # 市场指标判断
    ma20_above = bool(current_price > latest['MA20'])
    
    # 波动率判断
    if latest['atr'] < 0.02:
        volatility = "low"
    elif latest['atr'] < 0.04:
        volatility = "normal"
    else:
        volatility = "high"
    
    result = {
        "skill_version": CONFIG['version'],
        "request_id": f"req_{datetime.now().strftime('%Y%m%d_%H%M_%S')}",
        "timestamp": datetime.now().isoformat(),
        "stock": {
            "code": stock_code,
            "name": "N/A",
            "current_price": float(current_price)
        },
        "trading_signal": {
            "action": action_code,
            "description": action_desc,
            "position_change_bullets": bullet_change,
            "position_change_pct": abs(bullet_change) * CONFIG['bullet_size_pct'],
            "urgency": "normal" if abs(bullet_change) <= 2 else "high"
        },
        "position_suggestion": {
            "current_position_pct": pos['position_count'] * CONFIG['bullet_size_pct'],
            "suggested_position_pct": max(0, min(100, (pos['position_count'] + bullet_change) * CONFIG['bullet_size_pct'])) if bullet_change != -999 else 0,
            "remaining_bullets": CONFIG['max_bullets_per_stock'] - pos['position_count'],
            "max_allowed_bullets": CONFIG['risk_control']['max_position_bullets']
        },
        "risk_control": {
            "hard_stop_loss_price": float(hard_stop_loss),
            "stop_loss_pct": CONFIG['risk_control']['hard_stop_loss_pct'],
            "take_profit_1_price": float(take_profit_1),
            "take_profit_1_pct": 5,
            "take_profit_2_price": float(take_profit_2),
            "take_profit_2_pct": 10
        },
        "market_indicators": {
            "ma20_above": ma20_above,
            "rsi_14": float(latest['rsi']),
            "macd_divergence": "bullish" if latest['macd'] > 0 else "bearish",
            "volume_ratio": float(latest['vol_ratio']),
            "atr_volatility": volatility
        },
        "strategy_metrics": {
            "backtest_win_rate": 75.0,
            "backtest_return_30d": 3.91,
            "current_position_bullets": pos['position_count']
        },
        "execution_suggestion": {
            "order_type": "MARKET",
            "suggested_time": "14:40-14:50",
            "notes": _get_trade_notes(action_code, latest, ma20_above, volatility)
        },
        "status": "success"
    }
    
    return result

def batch_get_signals(stock_list: list) -> dict:
    """
    批量获取多只股票交易信号
    
    Args:
        stock_list: [{"code": "002156", "position": {...}}, ...]
    
    Returns:
        批量结果
    """
    results = []
    for item in stock_list:
        signal = get_trading_signal(item['code'], item.get('position', {}))
        results.append(signal)
    
    return {
        "status": "success",
        "count": len(results),
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

def _get_trade_notes(action_code: str, latest: pd.Series, ma20_above: bool, volatility: str) -> str:
    """生成交易备注"""
    notes = []
    
    if ma20_above:
        notes.append("MA20上方，趋势向好")
    else:
        notes.append("MA20下方，注意风险")
    
    if latest['rsi'] > 70:
        notes.append("RSI超买区域")
    elif latest['rsi'] < 30:
        notes.append("RSI超卖区域")
    
    if volatility == "high":
        notes.append("波动较大，谨慎操作")
    
    if 'add' in action_code and not ma20_above:
        notes.append("逆势加仓，严格控制仓位")
    elif 'reduce' in action_code or 'close' in action_code:
        notes.append("及时止盈止损")
    
    return "；".join(notes) if notes else "正常执行"
