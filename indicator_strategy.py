import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

class IndicatorStrategy:
    def __init__(self, symbol="000001", period="daily", interval="5"):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        
        # 仓位管理相关
        self.hold_positions = []  # 存储每次买入的详情 [{'price': p, 'time': t, 'shares': s}]
        self.max_add_times = 3    # 最多补仓3次（总共4仓位）
        self.max_hold_days = 20   # V21激进版：最长持仓20天
        self.trade_log = "indicator_trades_v2.txt"
        
        # 评分机制
        self.strategy_score = 0
        
        # RL 模型
        self.rl_model = None
        self.feature_cols = [
            'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
            'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio'
        ]

    def format_symbol(self, symbol):
        return f"sh{symbol}" if symbol.startswith(('60', '68', '90')) else f"sz{symbol}"

    def fetch_data(self):
        for _ in range(3):
            try:
                if self.period == "daily":
                    df = ak.stock_zh_a_daily(symbol=self.format_symbol(self.symbol), adjust="qfq")
                else:
                    df = ak.stock_zh_a_hist_min_em(symbol=self.symbol, period=self.interval, adjust="qfq")
                
                if df is not None and not df.empty:
                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'datetime'})
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    cols_map = {'开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}
                    df = df.rename(columns=cols_map)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
            except Exception as e:
                print(f"数据获取失败: {e}")
                time.sleep(2)
        return None

    def calculate_indicators(self, df):
        # 1. 均线系统 (MA)
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()

        # 2. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = exp1 - exp2
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        
        # 3. KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()

        # 4. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # 5. BOLL & 波动率
        df['boll_mid'] = df['close'].rolling(20).mean()
        df['boll_std'] = df['close'].rolling(20).std()
        df['boll_width'] = (df['boll_mid'] + 2*df['boll_std'] - (df['boll_mid'] - 2*df['boll_std'])) / df['boll_mid']

        # 6. ATR - 真实波幅 (衡量波动率)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean() / df['close']

        # 7. 价格相对位置
        df['price_ma20_ratio'] = (df['close'] - df['MA20']) / df['MA20'] * 100

        # 8. 量能确认
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma5']

        return df.fillna(0)

    def get_signals(self, df, current_time):
        """
        80% 胜率挑战版 (基于核心突破逻辑):
        
        策略核心: "大趋势下的放量起爆点"
        
        买入过滤:
        1. 趋势: MA5 > MA10 > MA20 (短中线多头排列)
        2. 突破: 价格突破过去 10 天的收盘价最高点 (确认起爆)
        3. 量能: 成交量 > 5日均量的 1.5 倍 (主力资金介入)
        4. 位置: RSI < 70 (避免极端超买)
        
        卖出/止盈:
        1. 3% 固定止盈 (确保极高胜率的核心)
        2. 5% 止损
        """
        if len(df) < 20: return "HOLD"
        
        last = df.iloc[-1]
        prev_10 = df.iloc[-11:-1]
        
        # 1. 突破信号
        is_breakout = last['close'] > prev_10['close'].max()
        # 2. 趋势信号
        is_perfect_ma = last['MA5'] > last['MA20'] and last['MA5'] > last['MA10']
        # 3. 量能爆发
        is_vol_spike = last['vol_ratio'] > 1.5
        
        # 持仓逻辑
        if self.hold_positions:
            total_cost = sum(p['price'] * p['shares'] for p in self.hold_positions)
            total_shares = sum(p['shares'] for p in self.hold_positions)
            avg_cost = total_cost / total_shares
            current_profit_pct = (last['close'] - avg_cost) / avg_cost * 100
            
            if current_profit_pct >= 3.0: return "TAKE_PROFIT"
            if current_profit_pct <= -5.0: return "STOP_LOSS"
            
            first_buy_time = self.hold_positions[0]['time']
            if (current_time - first_buy_time).days >= self.max_hold_days:
                return "TIMEOUT_EXIT"
            
            # 补仓逻辑 (满足买入条件且未满仓)
            if is_breakout and is_perfect_ma and is_vol_spike and last['rsi'] < 70:
                if len(self.hold_positions) <= self.max_add_times:
                    return "ADD_POSITION"

        # 买入逻辑
        if not self.hold_positions:
            if is_breakout and is_perfect_ma and is_vol_spike and last['rsi'] < 70:
                return "BUY"
                
        return "HOLD"

    def backtest(self, days=365):
        print(f"\n--- 开始补仓+限时策略回测: {self.symbol} ({self.period}) ---")
        df = self.fetch_data()
        if df is None: return
        
        if self.period == "daily":
            start_date = datetime.now() - pd.Timedelta(days=days)
            df = df[df['datetime'] >= start_date]
            
        df = self.calculate_indicators(df)
        
        cash = 20000 # 初始资金调高，方便多次补仓
        initial_cash = cash
        total_trade_count = 0
        total_hold_time = 0
        win_count = 0 
        self.strategy_score = 0 # 重置分数
        
        for i in range(60, len(df)): 
            sub_df = df.iloc[:i+1]
            curr = sub_df.iloc[-1]
            dt = curr['datetime']
            price = curr['close']
            
            signal = self.get_signals(sub_df, dt)
            
            # 处理买入/补仓
            if (signal == "BUY" or signal == "ADD_POSITION") and len(self.hold_positions) <= self.max_add_times:
                buy_amount = 5000
                if cash >= buy_amount:
                    shares = buy_amount // price
                    cash -= shares * price
                    self.hold_positions.append({'price': price, 'time': dt, 'shares': shares})
                    action_name = "初始买入" if signal == "BUY" else f"第{len(self.hold_positions)-1}次补仓"
                    
                    # 补仓奖励
                    if signal == "ADD_POSITION":
                        self.strategy_score += 1
                        
                    print(f"[{dt.date()}] {action_name}: {price:.2f} | RSI: {curr['rsi']:.1f} | 评分: {self.strategy_score}")

            # 处理卖出/清仓/止盈止损/超时
            elif signal in ["SELL", "TAKE_PROFIT", "STOP_LOSS", "TIMEOUT_EXIT"] and self.hold_positions:
                had_add_position = len(self.hold_positions) > 1
                total_shares = sum(p['shares'] for p in self.hold_positions)
                first_buy_time = self.hold_positions[0]['time']
                
                total_cost = sum(p['price'] * p['shares'] for p in self.hold_positions)
                revenue = total_shares * price
                profit_amount = revenue - total_cost
                profit_pct = profit_amount / total_cost * 100
                duration = (dt - first_buy_time).days
                
                # --- 惩罚与奖励逻辑 ---
                if signal == "TAKE_PROFIT":
                    self.strategy_score += 5
                elif signal == "STOP_LOSS":
                    self.strategy_score -= 5
                elif signal == "TIMEOUT_EXIT":
                    self.strategy_score -= 3
                
                if profit_amount > 0:
                    win_count += 1
                    if had_add_position: # 补仓后盈利奖励
                        self.strategy_score += 5
                else:
                    if abs(profit_pct) > 5: # 亏损 > 5% 惩罚
                        self.strategy_score -= 5
                    else: # 亏损 < 5% 惩罚
                        self.strategy_score -= 3
                
                cash += revenue
                total_trade_count += 1
                total_hold_time += duration
                
                reason_map = {
                    "SELL": "指标卖出", 
                    "TAKE_PROFIT": "止盈清仓", 
                    "STOP_LOSS": "止损清仓",
                    "TIMEOUT_EXIT": "超时清仓"
                }
                reason = reason_map.get(signal, "其他")
                print(f"[{dt.date()}] 【{reason}】: {price:.2f}, 收益: {profit_pct:.2f}%, 评分: {self.strategy_score}")
                self.hold_positions = []

        # 计算总结指标
        current_hold_value = sum(p['shares'] * df.iloc[-1]['close'] for p in self.hold_positions)
        final_value = cash + current_hold_value
        total_return_pct = (final_value-initial_cash)/initial_cash*100
        avg_hold_time = total_hold_time / total_trade_count if total_trade_count > 0 else 0
        win_rate = (win_count / total_trade_count * 100) if total_trade_count > 0 else 0
        
        print("\n" + "="*40)
        print(f"高胜率优化策略总结 ({self.symbol})")
        print(f"累计收益率: {total_return_pct:.2f}%")
        print(f"策略胜率: {win_rate:.2f}%")
        print(f"策略综合评分: {self.strategy_score}")
        print(f"平均持仓时间: {avg_hold_time:.2f} 天")
        print(f"总清仓次数: {total_trade_count}")
        print("="*40)

    def backtest_rl(self, model_path):
        """强化学习模型回测 (V2增强版: 扩展观测空间+优化评分策略)"""
        from stable_baselines3 import PPO
        print(f"\n=== 开始强化学习策略回测 (V2增强版): {self.symbol} ===")
        print(f"子弹规则: 总数10颗 | 单颗10%仓位 | 单次加减仓最多3颗")
        print(f"观测空间: 14个技术指标 + 6个状态特征")
        
        try:
            self.rl_model = PPO.load(model_path)
        except:
            print(f"错误：无法加载模型 {model_path}")
            return

        df = self.fetch_data()
        if df is None: return
        df = self.calculate_indicators(df)
        
        # 仅回测 2026 年至今
        test_df = df[df['datetime'] >= '2026-01-01'].copy()
        if test_df.empty:
            print("2026年至今无数据")
            return

        cash = 20000
        initial_cash = cash
        total_trade_count = 0
        win_count = 0
        self.strategy_score = 0
        self.hold_positions = []
        self.prev_floating_profit = 0
        self.max_trade_profit = 0
        self.consecutive_profit_days = 0
        self.last_action_dt = test_df.iloc[0]['datetime'] - pd.Timedelta(days=100)
        self.last_add_price = 0  # 最近一次加仓价格
        self.consecutive_add_count = 0  # 连续加仓次数，3次后强制减仓
        self.stop_loss_cooldown = 0  # 止损冷静期
        bullet_size_val = initial_cash * 0.1
        max_bullets = 9  # 9颗封顶
        max_action_bullets = 3
        
        feature_cols_v2 = [
            'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
            'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio',
            'boll_width', 'atr', 'price_ma20_ratio'
        ]

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            dt = row['datetime']
            price = row['close']
            
            # --- 动态评价系统 (同步 RL 环境 V2) ---
            if self.hold_positions:
                num_bullets = len(self.hold_positions)
                avg_cost_price = sum(p['price'] for p in self.hold_positions) / num_bullets
                stock_return_pct = (price - avg_cost_price) / avg_cost_price * 100
                total_cost_val = num_bullets * bullet_size_val
                total_shares = sum(bullet_size_val / p['price'] for p in self.hold_positions)
                current_value = total_shares * price
                current_pnl = current_value - total_cost_val
                
                # A. 子弹耗尽惩罚 (V5优化版：鼓励上升趋势加仓，轻度控制满仓)
                usage_ratio = num_bullets / max_bullets
                if usage_ratio >= 1.0:
                    self.strategy_score -= 4.0
                elif usage_ratio >= 0.9:
                    self.strategy_score -= 0.8
                
                # B. 持仓成本 (V21激进版：减少惩罚)
                self.strategy_score -= 0.01 * num_bullets
                
                # C. 趋势对齐奖励
                if price > row['MA20']:
                    self.strategy_score += 0.15
                
                # D. 利润增长奖励 (基于股票收益率计算)
                pnl_change = stock_return_pct - self.prev_floating_profit
                if pnl_change > 0:
                    self.strategy_score += pnl_change * 2.5
                    self.consecutive_profit_days += 1
                    if self.consecutive_profit_days >= 2:
                        self.strategy_score += 0.3 * self.consecutive_profit_days
                else:
                    self.strategy_score += pnl_change * 0.5
                    self.consecutive_profit_days = max(0, self.consecutive_profit_days - 1)
                
                # E. 加仓后大涨额外奖励 (V21激进版：提高奖励门槛)
                if self.last_add_price > 0 and price > self.last_add_price:
                    add_gain_pct = (price - self.last_add_price) / self.last_add_price * 100
                    if add_gain_pct >= 3.0:
                        extra_reward = (add_gain_pct - 3.0) * 2.0
                        self.strategy_score += extra_reward
                        if add_gain_pct >= 6.0:
                            self.strategy_score += 3.0
                        if add_gain_pct >= 10.0:
                            self.strategy_score += 8.0
                        self.last_add_price = 0
                
                # F. 浮盈回撤惩罚 (最大回撤控制在6%以内)
                self.max_trade_profit = max(self.max_trade_profit, current_pnl)
                if self.max_trade_profit > 0:
                    drawdown = (self.max_trade_profit - current_pnl) / total_cost_val * 100
                    if drawdown > 6:
                        self.strategy_score -= drawdown * 0.25
                
                # G. 持仓天数加权奖励/惩罚 (V21激进版)
                # 目标持仓: 1-5天(最佳) / 6-10天(理想) / 11-15天(正常) / 16-20天(惩罚)
                hold_days = (dt - self.hold_positions[0]['time']).days
                if hold_days <= 5:
                    self.strategy_score += 0.4
                elif hold_days <= 10:
                    self.strategy_score += 0.2
                elif hold_days < 20:
                    self.strategy_score -= (hold_days - 10) * 0.2
                
                # H. 快速止盈奖励 (5天内盈利止盈额外奖励)
                if hold_days <= 5 and current_pnl > 0:
                    self.strategy_score += 2.0
                
                self.prev_floating_profit = stock_return_pct
            else:
                self.prev_floating_profit = 0
                self.max_trade_profit = 0
                self.consecutive_profit_days = 0

            # 1. 强制时间检查
            if self.hold_positions:
                if (dt - self.hold_positions[0]['time']).days >= self.max_hold_days:
                    num_bullets = len(self.hold_positions)
                    total_shares = sum(bullet_size_val / p['price'] for p in self.hold_positions)
                    total_cost = num_bullets * bullet_size_val
                    revenue = total_shares * price
                    profit_pct = (revenue - total_cost) / total_cost * 100
                    
                    if profit_pct < 0:
                        loss_val = abs(profit_pct)
                        if loss_val > 8: self.strategy_score -= 50   # 亏损>8%重罚
                        elif loss_val > 5: self.strategy_score -= 10  # 亏损5-8%
                        elif loss_val > 3: self.strategy_score -= 5   # 亏损3-5%
                        else: self.strategy_score -= 2
                    else:
                        if profit_pct > 10: self.strategy_score += 8
                        elif profit_pct >= 5: self.strategy_score += 5
                        else: self.strategy_score += 2
                    
                    if profit_pct > 0: 
                        win_count += 1
                    
                    total_trade_count += 1
                    cash += revenue
                    print(f"[{dt.date()}] 【强制超时全清】: {price:.2f}, 收益: {profit_pct:.2f}%, 持仓: {num_bullets}颗, 评分: {self.strategy_score:.1f}")
                    self.hold_positions = []
                    self.prev_floating_profit = 0
                    self.max_trade_profit = 0
                    self.consecutive_profit_days = 0
                    continue

            # 1.5 动态止损检查 (浮亏超过6%立即止损)
            if self.hold_positions and floating_pnl_pct <= -6:
                num_bullets = len(self.hold_positions)
                total_cost = num_bullets * bullet_size_val
                total_shares = sum(bullet_size_val / p['price'] for p in self.hold_positions)
                revenue = total_shares * price
                profit_pct = (revenue - total_cost) / total_cost * 100
                
                self.strategy_score -= 30  # 止损惩罚
                self.strategy_score += profit_pct * 0.5
                
                if profit_pct > 0: win_count += 1
                total_trade_count += 1
                cash += revenue
                
                print(f"[{dt.date()}] 【止损】: {price:.2f}, 收益: {profit_pct:.2f}%, 持仓: {num_bullets}颗, 评分: {self.strategy_score:.1f}")
                self.hold_positions = []
                self.prev_floating_profit = 0
                self.max_trade_profit = 0
                self.consecutive_profit_days = 0
                self.stop_loss_cooldown = 5  # 止损后冷静期5天
                continue

            # 2. RL 动作预测 (V2 20维观测空间)
            tech_features = row[feature_cols_v2].values.astype(np.float32)
            
            current_count = len(self.hold_positions)
            if self.hold_positions:
                total_cost = current_count * bullet_size_val
                total_shares = sum(bullet_size_val / p['price'] for p in self.hold_positions)
                current_value = total_shares * price
                avg_cost_price = sum(p['price'] for p in self.hold_positions) / current_count
                
                floating_pnl_pct = (current_value - total_cost) / total_cost * 100
                hold_days = (dt - self.hold_positions[0]['time']).days
                avg_cost_ratio = (price - avg_cost_price) / avg_cost_price * 100
                
                if self.max_trade_profit > 0:
                    drawdown = (self.max_trade_profit - (current_value - total_cost)) / total_cost * 100
                else:
                    drawdown = 0
            else:
                floating_pnl_pct = 0
                hold_days = 0
                avg_cost_ratio = 0
                drawdown = 0
            
            # 计算追踪止损距离
                if hasattr(self, 'trailing_stop_price') and self.trailing_stop_price > 0:
                    trailing_stop_distance = (price - self.trailing_stop_price) / self.trailing_stop_price * 100
                else:
                    trailing_stop_distance = 0
                
                state_features = np.array([
                    current_count / max_bullets,
                    floating_pnl_pct / 10,
                    hold_days / self.max_hold_days,
                    avg_cost_ratio / 10,
                    drawdown / 5,
                    self.consecutive_profit_days / 5,
                    trailing_stop_distance / 5,
                    row['atr'] * 100
                ], dtype=np.float32)
            
            obs = np.concatenate([tech_features, state_features])
            action, _states = self.rl_model.predict(obs, deterministic=True)
            
            # 冷静期：止损后5天内只能持有/减仓
            if self.stop_loss_cooldown > 0:
                self.stop_loss_cooldown -= 1
                if action in [1, 2, 3]:  # 禁止加仓
                    action = 0  # 改为持有
            
            current_bullets = len(self.hold_positions)
            
            # G. 频繁交易惩罚
            days_since_last = (dt - self.last_action_dt).days
            if action != 0 and days_since_last < 1:
                self.strategy_score -= 0.3
            
            # 动作 1-3: 加仓 1-3 颗
            if action in [1, 2, 3]:
                add_bullets = action
                
                # J. 累计加仓次数限制（每轮持仓最多加仓3次）- 已禁用，回归V13
                add_count_limit_reached = False
                
                # I. 开仓限制：首次建仓最多5颗
                if current_bullets == 0 and add_bullets > 5:
                    self.strategy_score -= 2.0
                    add_bullets = 5
                
                available = max_bullets - current_bullets
                
                if available <= 0:
                    self.strategy_score -= 2.0
                elif add_count_limit_reached:
                    self.strategy_score -= 2.0  # 超过加仓次数限制
                else:
                    actual_add = min(add_bullets, available, max_action_bullets)
                    total_cost = actual_add * bullet_size_val
                    
                    if cash >= total_cost:
                        # J. 成功加仓，计数+1
                        self.consecutive_add_count += 1
                        
                        # 逆势加仓惩罚
                        if self.hold_positions and self.prev_floating_profit < 0:
                            self.strategy_score -= 0.5 * actual_add
                        
                        cash -= total_cost
                        for _ in range(actual_add):
                            self.hold_positions.append({'price': price, 'time': dt})
                        
                        self.strategy_score += 0.05 * actual_add
                        self.strategy_score -= 0.05 * actual_add
                        self.last_action_dt = dt
                        self.last_add_price = price  # 记录加仓价格，用于大涨奖励
                        print(f"[{dt.date()}] RL加{actual_add}颗: {price:.2f} | 仓位: {len(self.hold_positions)}/{max_bullets} | 评分: {self.strategy_score:.1f}")
                    else:
                        self.strategy_score -= 1.0

            # 动作 4-6: 减仓 1-3 颗
            elif action in [4, 5, 6] and self.hold_positions:
                reduce_bullets = action - 3
                actual_reduce = min(reduce_bullets, current_bullets)
                self.consecutive_add_count = 0  # 重置连续加仓计数
                total_reward = 0
                
                for _ in range(actual_reduce):
                    bullet = self.hold_positions.pop(0)
                    shares = bullet_size_val / bullet['price']
                    revenue = shares * price
                    pnl_pct = (revenue - bullet_size_val) / bullet_size_val * 100
                    cash += revenue
                    
                    if pnl_pct >= 3:
                        total_reward += 5
                    elif pnl_pct <= -5:
                        total_reward -= 5
                    elif pnl_pct > 0:
                        total_reward += pnl_pct / 3 * 1
                    else:
                        total_reward += pnl_pct / 5 * 1
                    
                    if pnl_pct > 0: win_count += 0.1
                    total_trade_count += 0.1
                
                self.strategy_score += total_reward
                self.strategy_score -= 0.05 * actual_reduce
                self.last_action_dt = dt
                print(f"[{dt.date()}] RL减{actual_reduce}颗: {price:.2f}, 收益分: {total_reward:.1f}, 持仓: {len(self.hold_positions)}颗, 评分: {self.strategy_score:.1f}")
                
                if not self.hold_positions:
                    self.prev_floating_profit = 0
                    self.max_trade_profit = 0

            # 动作 7: 全清
            elif action == 7 and self.hold_positions:
                num_bullets = len(self.hold_positions)
                total_cost = num_bullets * bullet_size_val
                total_shares = sum(bullet_size_val / p['price'] for p in self.hold_positions)
                revenue = total_shares * price
                profit_pct = (revenue - total_cost) / total_cost * 100
                
                if profit_pct >= 5.0:
                    clear_reward = 8
                elif profit_pct >= 3.0:
                    clear_reward = 5
                elif profit_pct <= -5.0:
                    clear_reward = -3
                elif profit_pct > 0:
                    clear_reward = profit_pct / 3 * 1
                else:
                    clear_reward = profit_pct * 0.8
                
                if profit_pct > 0 and num_bullets > 1:
                    clear_reward += 8
                
                self.strategy_score += clear_reward
                self.strategy_score -= 0.1
                self.last_action_dt = dt
                
                if profit_pct > 0: win_count += 1
                total_trade_count += 1
                cash += revenue
                
                print(f"[{dt.date()}] RL全清: {price:.2f}, 收益: {profit_pct:.2f}%, 清仓: {num_bullets}颗, 评分: {self.strategy_score:.1f}")
                self.hold_positions = []
                self.prev_floating_profit = 0
                self.max_trade_profit = 0
                self.consecutive_profit_days = 0

        final_value = cash + sum(bullet_size_val / p['price'] * test_df.iloc[-1]['close'] for p in self.hold_positions)
        print(f"\n===== RL V2增强版策略总结 ({self.symbol}) =====")
        print(f"累计收益: {(final_value-initial_cash)/initial_cash*100:.2f}%")
        print(f"策略胜率: {(win_count/total_trade_count*100) if total_trade_count>0 else 0:.2f}%")
        print(f"策略综合评分: {self.strategy_score:.1f}")
        print(f"总交易次数: {total_trade_count:.1f}")
        print(f"剩余持仓: {len(self.hold_positions)}颗")

if __name__ == "__main__":
    # 示例：使用强化学习模型回测 002156
    symbol = "002156"
    strategy = IndicatorStrategy(symbol=symbol, period="daily")
    
    # 指定训练好的模型路径 (由 train_rl_strategy.py 生成)
    model_path = f"ppo_trading_{symbol}.zip"
    
    if os.path.exists(model_path):
        strategy.backtest_rl(model_path)
    else:
        print(f"请先运行 python train_rl_strategy.py 为 {symbol} 训练模型")
