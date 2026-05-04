---
name: "rl-trading-live"
description: "强化学习量化交易实盘执行标准接口(V11版)。支持9颗子弹仓位管理、开仓≤5颗限制、趋势跟随让利润奔跑。当用户需要实盘交易信号、策略部署、自动交易时调用。"
---

# RL量化交易实盘标准接口 (V11稳定版)

基于 PPO 强化学习 + 9颗子弹动态仓位管理系统，兼容OpenClaw部署标准。

---

## 📋 标准API接口

### 1. `train_model(stock_code: str) -> str`
**功能**: 训练指定股票的强化学习模型

**参数**:
- `stock_code`: 股票代码，如 "002156"

**返回**: 模型文件路径

**调用示例**:
```python
from rl_trading_live import train_model
model_path = train_model("002156")
```

---

### 2. `get_trading_signal(stock_code: str, position_info: dict) -> dict`
**功能**: 获取实盘交易信号

**参数**:
```python
position_info = {
    "position_count": 0,      # 当前持仓子弹数 (0-9)
    "floating_pnl_pct": 0,    # 浮盈浮亏率 %
    "hold_days": 0,           # 持仓天数
    "avg_cost_price": 0,      # 平均持仓成本
    "entry_max_drawdown": 0,  # 入场后最大回撤 %
    "profit_run_days": 0      # 连续盈利天数
}
```

**返回**:
```python
{
    "timestamp": "2026-04-08 15:00:00",
    "stock_code": "002156",
    "current_price": 44.65,
    "action": "add_2",          # hold, add_1, add_2, add_3, reduce_1, reduce_2, reduce_3, close_all
    "action_desc": "加仓2颗",
    "bullet_size_pct": 10,     # 每颗子弹占比
    "suggest_position_change": 2,  # 建议仓位变化 +2
    "risk_indicators": {
        "ma20_above": True,
        "rsi": 58.3,
        "atr_volatility": "normal",
        "boll_width": 0.08
    },
    "risk_control": {
        "hard_stop_loss": 42.42,  # -5%止损
        "take_profit_1": 46.88,   # +5%止盈
        "take_profit_2": 49.12,   # +10%止盈
        "max_position_bullets": 9, # 最大9颗
        "max_open_position": 5,   # 开仓最多5颗
        "force_reduce_drawdown": 8 # 回撤8%强制减仓
    },
    "strategy_score": 162.2,
    "status": "success"
}
```

---

### 3. `batch_get_signals(stock_list: list) -> dict`
**功能**: 批量获取多只股票交易信号

---

### 4. `execute_trade(trade_signal: dict, trade_api: object) -> dict`
**功能**: 执行交易 (对接券商API)

---

## 🛡️ 标准风控规则 (V11版)

### 强制触发条件 (不可关闭)

| 规则 | 触发条件 | 动作 |
|------|---------|------|
| **硬止损** | 单笔浮亏 ≥ -5% | 100%全清 |
| **最大回撤** | 浮盈后回撤 ≥ 8% | 减仓至50% |
| **超时退出** | 持仓 ≥ 22个交易日 | 5个交易日内分批清仓 |
| **仓位上限** | 单票子弹数 ≥ 9颗 | 停止加仓 |
| **开仓限制** | 首次建仓 > 5颗 | 强制限制为5颗 |

---

## 📊 观测空间 (20维特征)

### 技术指标 (14维)
dif, dea, macd, k, d, rsi, MA5, MA10, MA20, MA60, vol_ratio, boll_width, atr, price_ma20_ratio

### 仓位状态 (6维)
position_count_norm, floating_pnl_pct, hold_days_norm, avg_cost_price_ratio, drawdown_since_entry, profit_run_days

---

## 📦 部署标准

### 目录结构

```
rl-trading-live/
├── __init__.py
├── config.py           # V11配置
├── data_fetcher.py    # 数据获取
├── signal_generator.py # 信号生成
├── risk_manager.py    # 风控
├── trade_executor.py  # 交易执行
├── models/            # 存放训练好的模型
│   └── ppo_trading_*.zip
└── logs/              # 交易日志
```

### V11核心配置

```python
CONFIG = {
    "version": "11.0.0",
    "max_bullets": 9,              # 9颗封顶
    "max_open_position": 5,       # 开仓最多5颗
    "bullet_size_pct": 10,
    "risk_control": {
        "hard_stop_loss_pct": -5,
        "max_drawdown_reduce_pct": 8,
        "max_hold_days": 22
    }
}
```

---

## 🚀 OpenClaw 部署

### 入口函数

```python
from rl_trading_live.openclaw_entry import execute

result = execute({
    "action": "signal",
    "stock_code": "002156",
    "position_info": {"position_count": 0},
    "dry_run": True
})
```

---

## ✅ 验证命令

```bash
# 1. 模块版本检查
python -c "import rl_trading_live; print(rl_trading_live.__version__)"

# 2. 健康检查
python -c "from rl_trading_live.openclaw_entry import execute; print(execute({'action':'health'})['status'])"

# 3. 完整示例
python rl_trading_live/example_usage.py
```

---

## 🎯 V11 vs 旧版本对比

| 特性 | V10及之前 | V11 |
|------|----------|-----|
| 子弹上限 | 10颗 | **9颗** |
| 开仓限制 | 无 | **≤5颗** |
| 胜率 | 48-75% | **90.48%** |
| 最大回撤容忍 | 3% | **8%** |
| 大涨奖励门槛 | 4% | **3%** |

---

**部署完成验证**：
```bash
python -c "import rl_trading_live; print(f'Version: {rl_trading_live.__version__}')"
```
