---
name: "rl-trading-live"
description: "强化学习量化交易实盘执行标准接口。支持股票训练、信号生成、风险控制、交易执行。当用户需要实盘交易信号、策略部署、自动交易时调用。"
---

# RL量化交易实盘标准接口 (V2)

基于 PPO 强化学习 + 10颗子弹动态仓位管理系统，兼容OpenClaw部署标准。

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
    "position_count": 0,      # 当前持仓子弹数 (0-10)
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
        "max_position_bullets": 6, # 最大6颗
        "force_reduce_drawdown": 8 # 回撤8%强制减仓
    },
    "strategy_score": 41.1
}
```

---

### 3. `batch_get_signals(stock_list: list) -> dict`
**功能**: 批量获取多只股票交易信号

**参数**:
```python
stock_list = [
    {"code": "002156", "position": position_info},
    {"code": "002079", "position": position_info}
]
```

---

### 4. `execute_trade(trade_signal: dict, trade_api: object) -> dict`
**功能**: 执行交易 (对接券商API)

**参数**:
- `trade_signal`: get_trading_signal返回的信号
- `trade_api`: 券商API对象，需实现以下方法:
  - `get_account_info()`
  - `place_order(code, direction, quantity)`
  - `get_position(code)`

**返回**: 执行结果

---

## 🛡️ 标准风控规则

### 强制触发条件 (不可关闭)

| 规则 | 触发条件 | 动作 |
|------|---------|------|
| **硬止损** | 单笔浮亏 ≥ -5% | 100%全清 |
| **最大回撤** | 浮盈后回撤 ≥ 8% | 减仓至剩余30% |
| **超时退出** | 持仓 ≥ 20个交易日 | 5个交易日内分批清仓 |
| **仓位上限** | 单票子弹数 ≥ 7颗 | 停止加仓 |
| **满仓禁止** | 子弹数 = 10颗 | 逆势惩罚翻倍 |

### 可选风控条件 (可配置)

| 规则 | 默认值 | 配置项 |
|------|-------|--------|
| 单日最大亏损 | -2% | `max_daily_loss_pct` |
| 连续亏损次数 | 3次 | `max_consecutive_losses` |
| 单票最大仓位 | 60% (6颗) | `max_single_position` |

---

## 📦 部署标准

### 1. 目录结构

```
rl-trading-live/
├── __init__.py
├── config.py           # 配置文件
├── data_fetcher.py    # 数据获取
├── signal_generator.py # 信号生成
├── risk_manager.py    # 风控
├── trade_executor.py  # 交易执行
├── models/            # 存放训练好的模型
│   ├── ppo_trading_002156.zip
│   └── ppo_trading_*.zip
└── logs/              # 交易日志
```

### 2. 配置文件 `config.py`

```python
CONFIG = {
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
        "signal_gen_time": "14:30"  # 每日14:30生成信号
    },
    "log_config": {
        "trade_log_path": "logs/trades/",
        "signal_log_path": "logs/signals/"
    }
}
```

---

## 🚀 OpenClaw 部署指南

### 1. 初始化 Skill

```python
# OpenClaw skill entrypoint
def execute(params: dict) -> dict:
    """
    OpenClaw 标准入口
    
    Args:
        params: {
            "action": "train|signal|batch|execute",
            "stock_code": "002156",
            "position_info": {...},
            "dry_run": True
        }
    """
    action = params.get("action", "signal")
    stock_code = params.get("stock_code")
    dry_run = params.get("dry_run", True)
    
    if action == "train":
        model_path = train_model(stock_code)
        return {"status": "success", "model_path": model_path}
    
    elif action == "signal":
        position_info = params.get("position_info", {})
        signal = get_trading_signal(stock_code, position_info)
        return signal
    
    elif action == "batch":
        stock_list = params.get("stock_list", [])
        results = batch_get_signals(stock_list)
        return {"status": "success", "results": results}
    
    else:
        return {"status": "error", "message": "Unknown action"}
```

### 2. 部署配置 (OpenClaw YAML)

```yaml
skill_id: rl-trading-live
version: 2.0.0
name: "RL强化学习量化交易"
description: "基于PPO+10颗子弹管理的A股量化交易系统"

entrypoint:
  module: rl_trading_live
  function: execute

parameters:
  - name: action
    type: string
    enum: [train, signal, batch, execute]
    required: true
    description: 执行动作类型

  - name: stock_code
    type: string
    required: true
    description: 股票代码，如002156

  - name: position_info
    type: object
    required: false
    description: 仓位信息对象

  - name: dry_run
    type: boolean  
    default: true
    description: 是否仅生成信号不实际交易

permissions:
  - network: true
  - file_write: logs/
  - model_load: models/*.zip

rate_limit:
  max_calls_per_day: 100
  max_concurrent: 5
```

---

## 📊 标准输出格式

### 单次信号响应示例

```json
{
    "skill_version": "2.0.0",
    "request_id": "req_20260408_1430_001",
    "timestamp": "2026-04-08T14:30:00+08:00",
    "stock": {
        "code": "002156",
        "name": "通富微电",
        "current_price": 44.65
    },
    "trading_signal": {
        "action": "add_2",
        "description": "加仓2颗",
        "position_change_bullets": 2,
        "position_change_pct": 20,
        "urgency": "normal"
    },
    "position_suggestion": {
        "current_position_pct": 20,
        "suggested_position_pct": 40,
        "remaining_bullets": 8,
        "max_allowed_bullets": 6
    },
    "risk_control": {
        "hard_stop_loss_price": 42.42,
        "stop_loss_pct": -5,
        "take_profit_1_price": 46.88,
        "take_profit_1_pct": 5,
        "take_profit_2_price": 49.12,
        "take_profit_2_pct": 10
    },
    "market_indicators": {
        "ma20_above": true,
        "rsi_14": 58.3,
        "macd_divergence": "none",
        "volume_ratio": 1.23,
        "atr_volatility": "normal"
    },
    "strategy_metrics": {
        "backtest_win_rate": 75.0,
        "backtest_return_30d": 3.91,
        "current_strategy_score": 41.1
    },
    "execution_suggestion": {
        "order_type": "MARKET",
        "suggested_time": "14:40-14:50",
        "notes": "MA20上方，趋势良好，建议分批执行"
    },
    "status": "success"
}
```

---

## 🔔 告警与监控

### 标准告警级别

| 级别 | 触发条件 | 通知方式 |
|------|---------|---------|
| **CRITICAL** | 触发硬止损/单日亏损>3% | 电话+短信+邮件 |
| **WARNING** | 回撤>5%/连续亏损2次 | 邮件+应用推送 |
| **INFO** | 交易信号生成/正常调仓 | 应用内通知 |
| **DEBUG** | 详细运行日志 | 仅日志 |

### 标准监控指标

```python
MONITOR_METRICS = [
    "daily_trade_count",           # 每日交易次数
    "daily_return_pct",            # 当日收益率
    "win_rate_rolling_30d",        # 30日滚动胜率
    "max_drawdown_rolling",        # 滚动最大回撤
    "bullet_utilization_rate",     # 子弹使用率
    "consecutive_losses",          # 连续亏损次数
    "strategy_score_daily"         # 策略每日评分
]
```

---

## ✅ Skill 验证清单

部署前验证：

- [ ] 所有依赖已安装 (stable-baselines3, akshare, numpy, pandas)
- [ ] 模型文件存在于 `models/` 目录
- [ ] `logs/` 目录可写
- [ ] 训练函数可正常运行并保存模型
- [ ] 信号生成函数返回标准格式
- [ ] 风控规则正确触发
- [ ] 回测结果符合预期 (胜率>60%, 最大回撤<15%)
- [ ] OpenClaw接口参数正确映射

---

## 📝 标准Python实现文件

### `__init__.py`

```python
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
```

---

## 🎯 标准使用流程

### 最小化使用

```python
from rl_trading_live import get_trading_signal

# 持仓为空时获取信号
signal = get_trading_signal("002156", position_info={
    "position_count": 0,
    "floating_pnl_pct": 0,
    "hold_days": 0,
    "avg_cost_price": 0,
    "entry_max_drawdown": 0,
    "profit_run_days": 0
})

print(f"建议动作: {signal['trading_signal']['description']}")
```

### 完整每日流程

```python
"""
每日14:30定时任务
"""
from rl_trading_live import get_trading_signal, check_stop_loss

# 1. 获取持仓信息 (从券商API)
positions = trade_api.get_positions()

# 2. 检查风控止损
check_stop_loss(positions)

# 3. 生成交易信号
for pos in positions:
    signal = get_trading_signal(pos['code'], pos_to_info(pos))
    
    # 4. 非观望则执行
    if signal['trading_signal']['action'] != 'hold':
        if not dry_run:
            execute_trade(signal, trade_api)
        
        # 5. 记录日志
        log_trade_signal(signal)

# 6. 发送每日报告
send_daily_report()
```

---

## 🔄 版本兼容性

| Skill版本 | 环境版本 | 说明 |
|----------|---------|------|
| 2.0.0 | Python 3.8+, gymnasium | 当前标准版，20维观测空间 |
| 1.x.x | 向后兼容 | 建议升级到2.0获取更佳风控效果 |

---

## 📂 技能文件结构

```
.trae/skills/rl-trading-live/
└── SKILL.md              # 技能文档

# Python包路径
rl_trading_live/
├── __init__.py           # 模块入口
├── config.py             # 配置文件
├── data_fetcher.py       # 数据获取
├── signal_generator.py   # 信号生成
├── risk_manager.py       # 风险管理
├── trade_executor.py     # 交易执行
├── openclaw_entry.py     # OpenClaw标准入口
├── example_usage.py      # 使用示例
└── requirements.txt      # 依赖清单
```

---

## ✅ 部署验证命令

```bash
# 1. 验证模块加载
python -c "import rl_trading_live; print(f'Version: {rl_trading_live.__version__}')"

# 2. 验证信号生成 (需先有模型)
python -c "
from rl_trading_live import get_trading_signal
result = get_trading_signal('002156', {})
print(f\"Status: {result['status']}\")
print(f\"Action: {result['trading_signal']['action']}\")
"

# 3. OpenClaw入口测试
python -c "
from rl_trading_live.openclaw_entry import execute
print('健康检查:', execute({'action': 'health'})['status'])
"

# 4. 运行完整示例
python rl_trading_live/example_usage.py
```

**部署检查清单**:

- [ ] Python 3.8+
- [ ] 已安装依赖: `pip install -r rl_trading_live/requirements.txt`
- [ ] 模型已训练并放入 `models/` 目录
- [ ] `logs/` 目录可写
- [ ] 信号生成正常返回20维观测
- [ ] 风控模块正确触发硬止损
- [ ] OpenClaw execute() 入口正常响应
