"""
RL Trading Live - 强化学习量化交易实盘标准接口
Version: 2.0.0
Compatible: OpenClaw Platform
"""

__version__ = "2.0.0"
__author__ = "RL Quant Team"

from .data_fetcher import fetch_stock_data, calculate_indicators
from .signal_generator import get_trading_signal, batch_get_signals, train_model
from .risk_manager import apply_risk_controls, check_stop_loss
from .trade_executor import execute_trade, execute_batch_trades
from .config import CONFIG

__all__ = [
    "train_model",
    "get_trading_signal", 
    "batch_get_signals",
    "execute_trade",
    "execute_batch_trades",
    "fetch_stock_data",
    "calculate_indicators",
    "apply_risk_controls",
    "check_stop_loss",
    "CONFIG"
]
