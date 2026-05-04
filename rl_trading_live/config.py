"""
配置文件
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "version": "2.0.0",
    "initial_capital": 100000,           # 初始资金
    "bullet_size_pct": 10,               # 每颗子弹10%
    "max_bullets_per_stock": 10,         # 单票最大子弹数
    "max_add_per_action": 3,             # 单次加仓上限
    "risk_control": {
        "hard_stop_loss_pct": -5,
        "max_drawdown_reduce_pct": 8,
        "max_hold_days": 20,
        "max_position_bullets": 6
    },
    "trading_hours": {
        "start": "09:30",
        "end": "15:00",
        "signal_gen_time": "14:30"
    },
    "model_dir": os.path.join(BASE_DIR, "models"),
    "log_config": {
        "trade_log_path": os.path.join(BASE_DIR, "logs/trades/"),
        "signal_log_path": os.path.join(BASE_DIR, "logs/signals/")
    },
    "feature_cols": [
        'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
        'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio',
        'boll_width', 'atr', 'price_ma20_ratio'
    ]
}

# 创建必要目录
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['log_config']['trade_log_path'], exist_ok=True)
os.makedirs(CONFIG['log_config']['signal_log_path'], exist_ok=True)
