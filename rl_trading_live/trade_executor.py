"""
交易执行模块
对接券商API
"""
import os
import json
from datetime import datetime
from typing import Optional, Any

from .config import CONFIG

class BaseTradeAPI:
    """
    券商API基类 - 继承此类实现具体对接
    """
    def get_account_info(self) -> dict:
        """获取账户信息"""
        raise NotImplementedError
    
    def place_order(self, code: str, direction: str, amount: int, price: Optional[float] = None) -> dict:
        """
        下单
        
        Args:
            code: 股票代码
            direction: 'BUY' 或 'SELL'
            amount: 股数 (100的整数倍)
            price: 价格，None为市价单
        
        Returns:
            {
                "order_id": str,
                "status": "success|pending|failed",
                "message": str
            }
        """
        raise NotImplementedError
    
    def get_position(self, code: str) -> dict:
        """获取单只股票持仓"""
        raise NotImplementedError
    
    def get_all_positions(self) -> list:
        """获取所有持仓"""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> dict:
        """撤单"""
        raise NotImplementedError

def execute_trade(signal: dict, trade_api: BaseTradeAPI, dry_run: bool = True) -> dict:
    """
    执行交易 (标准接口)
    
    Args:
        signal: get_trading_signal返回的信号
        trade_api: 券商API实例
        dry_run: 是否仅模拟不实际下单
    
    Returns:
        执行结果
    """
    stock_code = signal['stock']['code']
    current_price = signal['stock']['current_price']
    action = signal['trading_signal']['action']
    position_change = signal['trading_signal']['position_change_bullets']
    
    execution_result = {
        "signal": signal,
        "timestamp": datetime.now().isoformat(),
        "stock_code": stock_code,
        "action": action,
        "dry_run": dry_run,
        "orders": []
    }
    
    # 观望不操作
    if action == 'hold':
        execution_result['status'] = 'skipped'
        execution_result['message'] = '观望信号，不执行交易'
        return execution_result
    
    # 获取当前持仓
    try:
        position = trade_api.get_position(stock_code)
        current_shares = position.get('shares', 0)
        account = trade_api.get_account_info()
        available_cash = account.get('available_cash', 0)
    except Exception as e:
        execution_result['status'] = 'error'
        execution_result['message'] = f'获取持仓/账户信息失败: {str(e)}'
        return execution_result
    
    # 计算每颗子弹对应的股数
    account_value = account.get('total_assets', CONFIG['initial_capital'])
    per_bullet_value = account_value * CONFIG['bullet_size_pct'] / 100
    
    # 计算股数 (100股为1手，向上取整到最近的100)
    shares_per_bullet = int(per_bullet_value / current_price)
    shares_per_bullet = max(100, (shares_per_bullet // 100) * 100)
    
    # ========== 买入/加仓 ==========
    if action in ['add_1', 'add_2', 'add_3']:
        bullets_to_buy = int(action.split('_')[1])
        total_shares = shares_per_bullet * bullets_to_buy
        est_cost = total_shares * current_price * 1.001  # 估算含手续费
        
        if available_cash < est_cost:
            execution_result['status'] = 'failed'
            execution_result['message'] = f'资金不足: 需要{est_cost:.2f}, 可用{available_cash:.2f}'
            return execution_result
        
        if not dry_run:
            order_result = trade_api.place_order(
                code=stock_code,
                direction='BUY',
                amount=total_shares,
                price=None  # 市价单
            )
            execution_result['orders'].append(order_result)
        else:
            execution_result['orders'].append({
                "order_id": f"SIM_{int(datetime.now().timestamp())}",
                "status": "simulated",
                "direction": "BUY",
                "shares": total_shares,
                "est_cost": est_cost,
                "per_bullet_shares": shares_per_bullet
            })
        
        execution_result['status'] = 'executed' if not dry_run else 'simulated'
        execution_result['message'] = f'加仓{bullets_to_buy}颗, 预计{total_shares}股'
    
    # ========== 卖出/减仓 ==========
    elif action in ['reduce_1', 'reduce_2', 'reduce_3']:
        bullets_to_sell = int(action.split('_')[1])
        shares_to_sell = shares_per_bullet * bullets_to_sell
        
        if current_shares <= 0:
            execution_result['status'] = 'failed'
            execution_result['message'] = '无持仓可减'
            return execution_result
        
        # 不能卖超过持仓
        actual_shares = min(shares_to_sell, current_shares)
        
        if not dry_run:
            order_result = trade_api.place_order(
                code=stock_code,
                direction='SELL',
                amount=actual_shares,
                price=None
            )
            execution_result['orders'].append(order_result)
        else:
            execution_result['orders'].append({
                "order_id": f"SIM_{int(datetime.now().timestamp())}",
                "status": "simulated",
                "direction": "SELL",
                "shares": actual_shares,
                "est_revenue": actual_shares * current_price * 0.999
            })
        
        execution_result['status'] = 'executed' if not dry_run else 'simulated'
        execution_result['message'] = f'减仓{bullets_to_sell}颗, 卖出{actual_shares}股'
    
    # ========== 全清 ==========
    elif action == 'close_all':
        if current_shares <= 0:
            execution_result['status'] = 'skipped'
            execution_result['message'] = '无持仓可清'
            return execution_result
        
        if not dry_run:
            order_result = trade_api.place_order(
                code=stock_code,
                direction='SELL',
                amount=current_shares,
                price=None
            )
            execution_result['orders'].append(order_result)
        else:
            execution_result['orders'].append({
                "order_id": f"SIM_{int(datetime.now().timestamp())}",
                "status": "simulated",
                "direction": "SELL",
                "shares": current_shares,
                "est_revenue": current_shares * current_price * 0.999
            })
        
        execution_result['status'] = 'executed' if not dry_run else 'simulated'
        execution_result['message'] = f'全清, 卖出{current_shares}股'
    
    # 写日志
    _log_trade_execution(execution_result)
    
    return execution_result

def execute_batch_trades(signals: list, trade_api: BaseTradeAPI, dry_run: bool = True) -> dict:
    """
    批量执行交易
    """
    results = []
    for signal in signals:
        result = execute_trade(signal, trade_api, dry_run)
        results.append(result)
    
    success_count = sum(1 for r in results if r['status'] in ['executed', 'simulated'])
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    return {
        "status": "completed",
        "total": len(signals),
        "success": success_count,
        "failed": failed_count,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

def _log_trade_execution(result: dict):
    """记录交易执行日志"""
    log_path = os.path.join(
        CONFIG['log_config']['trade_log_path'],
        f"trade_{datetime.now().strftime('%Y%m%d')}.jsonl"
    )
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception:
        pass
