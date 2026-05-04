"""V23版本强化学习策略回测 - 完全匹配训练环境逻辑"""
import akshare as ak
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from indicator_strategy import IndicatorStrategy

def backtest_v23(symbol="002156"):
    """V23优化版回测 - 完全同步训练环境逻辑"""
    print(f"\n===== RL V23优化版策略回测: {symbol} =====")
    print(f"子弹规则: 9颗封顶 | 单颗10%仓位 | 单次加减仓最多3颗")
    print(f"止损规则: 5%固定止损 + 动态ATR追踪止损")
    print(f"持仓周期: 最长15天")
    
    # 加载模型
    model_path = f"ppo_trading_{symbol}.zip"
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"错误：无法加载模型 {model_path}: {e}")
        return
    
    # 获取数据
    strategy = IndicatorStrategy(symbol=symbol)
    df = strategy.fetch_data()
    if df is None: return
    df = strategy.calculate_indicators(df)
    
    # 回测区间：2026年至今
    test_df = df[df['datetime'] >= '2026-01-01'].copy()
    if test_df.empty:
        print("2026年至今无数据")
        return
    
    print(f"\n回测区间: {test_df['datetime'].min().date()} 至 {test_df['datetime'].max().date()}")
    
    # 初始化参数（完全同步训练环境）
    cash = 20000
    initial_cash = cash
    hold_positions = []  # [{'price': p, 'step': i}]
    max_bullets = 9
    bullet_size_val = initial_cash * 0.1
    max_add_per_action = 3
    max_hold_days = 15
    trailing_stop_enabled = True  # 启用追踪止损
    
    # 状态变量
    prev_floating_profit = 0
    max_trade_profit = 0
    consecutive_profit_days = 0
    last_action_step = -100
    last_add_price = 0
    trailing_stop_price = 0
    trailing_stop_initialized = False
    
    # 统计变量
    total_trades = 0
    win_count = 0
    strategy_score = 0
    trade_log = []
    
    # 特征列（14个技术指标）
    feature_cols = [
        'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
        'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio',
        'boll_width', 'atr', 'price_ma20_ratio'
    ]
    
    # 开始回测
    for step in range(len(test_df)):
        row = test_df.iloc[step]
        price = row['close']
        atr_value = row['atr']
        dt = row['datetime']
        
        # ==================== 持仓状态评估 (V23评分系统) ====================
        if hold_positions:
            num_bullets = len(hold_positions)
            total_cost = num_bullets * bullet_size_val
            total_shares = sum(bullet_size_val / p['price'] for p in hold_positions)
            current_value = total_shares * price
            current_pnl = current_value - total_cost
            current_pnl_pct = current_pnl / total_cost * 100
            
            # A. 降低满仓惩罚 (V23优化)
            usage_ratio = num_bullets / max_bullets
            if usage_ratio >= 1.0:
                strategy_score -= 1.5
            elif usage_ratio >= 0.9:
                strategy_score -= 0.3
            
            # B. 降低持仓成本惩罚 (V23优化)
            strategy_score -= 0.005 * num_bullets
            
            # C. 增强趋势对齐奖励
            if price > row['MA20']:
                strategy_score += 0.2
                if row['MA5'] > row['MA10'] and row['MA10'] > row['MA20']:
                    strategy_score += 0.15
            
            # D. 利润增长奖励 - 增强版
            pnl_change = current_pnl_pct - prev_floating_profit
            if pnl_change > 0:
                strategy_score += pnl_change * 3.0
                consecutive_profit_days += 1
                if consecutive_profit_days >= 2:
                    strategy_score += 0.4 * consecutive_profit_days
            else:
                strategy_score += pnl_change * 0.3
                consecutive_profit_days = max(0, consecutive_profit_days - 1)
            
            # E. 小额盈利即时奖励 (V23新增)
            if current_pnl_pct > 1.0 and current_pnl_pct < 5.0:
                strategy_score += 0.5
            
            # F. 加仓后上涨奖励
            if last_add_price > 0 and price > last_add_price:
                add_gain_pct = (price - last_add_price) / last_add_price * 100
                if add_gain_pct >= 2.0:
                    extra_reward = (add_gain_pct - 2.0) * 2.5
                    strategy_score += extra_reward
                    if add_gain_pct >= 5.0:
                        strategy_score += 4.0
                    if add_gain_pct >= 8.0:
                        strategy_score += 10.0
                    last_add_price = 0
            
            # G. 浮盈回撤惩罚 - 非线性加重
            max_trade_profit = max(max_trade_profit, current_pnl)
            if max_trade_profit > 0:
                drawdown = (max_trade_profit - current_pnl) / total_cost * 100
                if drawdown > 5:
                    strategy_score -= drawdown ** 1.2 * 0.15
            
            # H. 持仓天数优化 (V23)
            hold_days = step - hold_positions[0]['step']
            if hold_days <= 4:
                strategy_score += 0.5
            elif hold_days <= 8:
                strategy_score += 0.25
            elif hold_days <= 12:
                strategy_score += 0.1
            elif hold_days < max_hold_days:
                strategy_score -= (hold_days - 12) * 0.15
            
            # I. 快速止盈奖励增强
            if hold_days <= 4 and current_pnl > 0:
                strategy_score += 3.0
            
            # J. 更新追踪止损
            if trailing_stop_enabled and not trailing_stop_initialized:
                trailing_stop_price = price * (1 - 0.03)  # 3%初始止损
                trailing_stop_initialized = True
            
            # K. 超时强制清仓
            if hold_days >= max_hold_days:
                # 超时清仓评分
                if current_pnl_pct < 0:
                    loss_val = abs(current_pnl_pct)
                    if loss_val > 6:
                        strategy_score -= 30
                    elif loss_val > 4:
                        strategy_score -= 8
                    elif loss_val > 2:
                        strategy_score -= 3
                    else:
                        strategy_score -= 1
                else:
                    if current_pnl_pct > 8:
                        strategy_score += 10
                    elif current_pnl_pct >= 4:
                        strategy_score += 6
                    elif current_pnl_pct >= 2:
                        strategy_score += 3
                    else:
                        strategy_score += 1
                
                total_trades += 1
                if current_pnl_pct > 0:
                    win_count += 1
                
                cash += current_value
                trade_log.append({
                    'date': dt.date(),
                    'action': 'TIMEOUT',
                    'price': price,
                    'pnl_pct': current_pnl_pct,
                    'bullets': num_bullets,
                    'score': strategy_score
                })
                print(f"[{dt.date()}] 【超时清仓】: {price:.2f}, 收益: {current_pnl_pct:.2f}%, 持仓: {num_bullets}颗")
                
                hold_positions = []
                prev_floating_profit = 0
                max_trade_profit = 0
                consecutive_profit_days = 0
                trailing_stop_price = 0
                trailing_stop_initialized = False
                continue
            
            prev_floating_profit = current_pnl_pct
            
            # ==================== 动态止损检查 (V23) ====================
            avg_cost_price = sum(p['price'] for p in hold_positions) / num_bullets
            
            # 固定止损检查
            if current_pnl_pct <= -5:
                strategy_score -= 15  # V23: 降低止损惩罚
                strategy_score += current_pnl_pct * 0.5
                
                total_trades += 1
                if current_pnl_pct > 0:
                    win_count += 1
                
                cash += current_value
                trade_log.append({
                    'date': dt.date(),
                    'action': 'STOP_LOSS',
                    'price': price,
                    'pnl_pct': current_pnl_pct,
                    'bullets': num_bullets,
                    'score': strategy_score
                })
                print(f"[{dt.date()}] 【止损】: {price:.2f}, 收益: {current_pnl_pct:.2f}%, 持仓: {num_bullets}颗")
                
                hold_positions = []
                prev_floating_profit = 0
                max_trade_profit = 0
                consecutive_profit_days = 0
                trailing_stop_price = 0
                trailing_stop_initialized = False
                continue
            
            # 追踪止损检查
            if trailing_stop_enabled and trailing_stop_price > 0:
                # 更新追踪止损价
                current_pnl_pct_check = (price - avg_cost_price) / avg_cost_price * 100
                if current_pnl_pct_check >= 3.0:
                    new_stop = price * 0.97  # 3%追踪止损
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                if price <= trailing_stop_price:
                    strategy_score -= 15
                    strategy_score += current_pnl_pct * 0.5
                    
                    total_trades += 1
                    if current_pnl_pct > 0:
                        win_count += 1
                    
                    cash += current_value
                    trade_log.append({
                        'date': dt.date(),
                        'action': 'TRAILING_STOP',
                        'price': price,
                        'pnl_pct': current_pnl_pct,
                        'bullets': num_bullets,
                        'score': strategy_score
                    })
                    print(f"[{dt.date()}] 【追踪止损】: {price:.2f}, 收益: {current_pnl_pct:.2f}%, 持仓: {num_bullets}颗")
                    
                    hold_positions = []
                    prev_floating_profit = 0
                    max_trade_profit = 0
                    consecutive_profit_days = 0
                    trailing_stop_price = 0
                    trailing_stop_initialized = False
                    continue
        else:
            prev_floating_profit = 0
            max_trade_profit = 0
            consecutive_profit_days = 0
            trailing_stop_price = 0
            trailing_stop_initialized = False
        
        # ==================== RL动作预测 ====================
        # 构建观测向量 (14技术指标 + 8状态特征 = 22维)
        tech_features = row[feature_cols].values.astype(np.float32)
        
        current_count = len(hold_positions)
        if hold_positions:
            total_cost = current_count * bullet_size_val
            total_shares = sum(bullet_size_val / p['price'] for p in hold_positions)
            current_value = total_shares * price
            avg_cost_price = sum(p['price'] for p in hold_positions) / current_count
            
            floating_pnl_pct = (current_value - total_cost) / total_cost * 100
            hold_days = step - hold_positions[0]['step']
            avg_cost_ratio = (price - avg_cost_price) / avg_cost_price * 100
            
            if max_trade_profit > 0:
                drawdown = (max_trade_profit - (current_value - total_cost)) / total_cost * 100
            else:
                drawdown = 0
        else:
            floating_pnl_pct = 0
            hold_days = 0
            avg_cost_ratio = 0
            drawdown = 0
        
        # 计算追踪止损距离
        if trailing_stop_price > 0:
            trailing_stop_distance = (price - trailing_stop_price) / trailing_stop_price * 100
        else:
            trailing_stop_distance = 0
        
        state_features = np.array([
            current_count / max_bullets,
            floating_pnl_pct / 10,
            hold_days / max_hold_days,
            avg_cost_ratio / 10,
            drawdown / 5,
            consecutive_profit_days / 5,
            trailing_stop_distance / 5,
            atr_value * 100
        ], dtype=np.float32)
        
        obs = np.concatenate([tech_features, state_features])
        action, _states = model.predict(obs, deterministic=True)
        
        # ==================== 动作执行 ====================
        # 降低频繁交易惩罚
        days_since_last = step - last_action_step
        if action != 0 and days_since_last < 1:
            strategy_score -= 0.1
        
        current_bullets = len(hold_positions)
        
        # 动作 1-3: 加仓
        if action in [1, 2, 3]:
            add_bullets = action
            
            # 开仓限制放宽 (V23: 最多7颗)
            if current_bullets == 0 and add_bullets > 7:
                strategy_score -= 1.0
                add_bullets = 7
            
            available = max_bullets - current_bullets
            
            if available <= 0:
                strategy_score -= 1.0
            else:
                actual_add = min(add_bullets, available, max_add_per_action)
                total_cost = actual_add * bullet_size_val
                
                if cash >= total_cost:
                    cash -= total_cost
                    for _ in range(actual_add):
                        hold_positions.append({'price': price, 'step': step})
                    
                    strategy_score += 0.1 * actual_add
                    strategy_score -= 0.03 * actual_add
                    last_action_step = step
                    last_add_price = price
                    
                    # 初始化追踪止损
                    if trailing_stop_enabled and not trailing_stop_initialized:
                        trailing_stop_price = price * 0.97
                        trailing_stop_initialized = True
                    
                    print(f"[{dt.date()}] RL加{actual_add}颗: {price:.2f} | 仓位: {len(hold_positions)}/{max_bullets}")
                else:
                    strategy_score -= 0.5
        
        # 动作 4-6: 减仓
        elif action in [4, 5, 6] and hold_positions:
            reduce_bullets = action - 3
            actual_reduce = min(reduce_bullets, current_bullets)
            total_reward = 0
            
            for _ in range(actual_reduce):
                bullet = hold_positions.pop(0)
                shares = bullet_size_val / bullet['price']
                revenue = shares * price
                pnl_pct = (revenue - bullet_size_val) / bullet_size_val * 100
                cash += revenue
                
                if pnl_pct >= 2:
                    total_reward += 6
                elif pnl_pct >= 1:
                    total_reward += 3
                elif pnl_pct <= -3:
                    total_reward -= 3
                elif pnl_pct > 0:
                    total_reward += pnl_pct * 1.5
                else:
                    total_reward += pnl_pct * 0.5
            
            strategy_score += total_reward
            strategy_score -= 0.03 * actual_reduce
            last_action_step = step
            
            if not hold_positions:
                prev_floating_profit = 0
                max_trade_profit = 0
                trailing_stop_price = 0
                trailing_stop_initialized = False
        
        # 动作 7: 全清
        elif action == 7 and hold_positions:
            num_bullets = len(hold_positions)
            total_cost = num_bullets * bullet_size_val
            total_shares = sum(bullet_size_val / p['price'] for p in hold_positions)
            revenue = total_shares * price
            profit_pct = (revenue - total_cost) / total_cost * 100
            
            # V23清仓评分
            if profit_pct >= 5.0:
                clear_reward = 10
            elif profit_pct >= 3.0:
                clear_reward = 6
            elif profit_pct >= 1.0:
                clear_reward = 3
            elif profit_pct <= -4.0:
                clear_reward = -2
            else:
                clear_reward = profit_pct * 1.0
            
            if profit_pct > 0 and num_bullets > 1:
                clear_reward += 2 * num_bullets
            
            strategy_score += clear_reward
            strategy_score -= 0.05
            last_action_step = step
            
            total_trades += 1
            if profit_pct > 0:
                win_count += 1
            
            cash += revenue
            trade_log.append({
                'date': dt.date(),
                'action': 'MANUAL',
                'price': price,
                'pnl_pct': profit_pct,
                'bullets': num_bullets,
                'score': strategy_score
            })
            print(f"[{dt.date()}] RL全清: {price:.2f}, 收益: {profit_pct:.2f}%, 清仓: {num_bullets}颗")
            
            hold_positions = []
            prev_floating_profit = 0
            max_trade_profit = 0
            consecutive_profit_days = 0
            trailing_stop_price = 0
            trailing_stop_initialized = False
    
    # ==================== 输出结果 ====================
    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash * 100
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    print("\n" + "="*50)
    print(f"RL V23优化版策略总结 ({symbol})")
    print("="*50)
    print(f"初始资金: {initial_cash:.2f}")
    print(f"最终资产: {final_value:.2f}")
    print(f"累计收益率: {total_return:.2f}%")
    print(f"策略胜率: {win_rate:.2f}%")
    print(f"策略综合评分: {strategy_score:.1f}")
    print(f"总交易次数: {total_trades}")
    print(f"盈利次数: {win_count}")
    print("="*50)
    
    # 详细交易记录
    if trade_log:
        print("\n交易记录详情:")
        print("-"*50)
        print(f"{'日期':<12} {'操作':<12} {'价格':<8} {'收益':<10} {'仓位':<6}")
        print("-"*50)
        for trade in trade_log:
            action_name = {
                'MANUAL': '手动清仓',
                'STOP_LOSS': '止损',
                'TIMEOUT': '超时清仓',
                'TRAILING_STOP': '追踪止损'
            }.get(trade['action'], trade['action'])
            print(f"{trade['date']} {action_name:<12} {trade['price']:<8.2f} {trade['pnl_pct']:<10.2f}% {trade['bullets']:<6}颗")
    
    return {
        'return': total_return,
        'win_rate': win_rate,
        'trades': total_trades,
        'score': strategy_score
    }

if __name__ == "__main__":
    backtest_v23("002156")
