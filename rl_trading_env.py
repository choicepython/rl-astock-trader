import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=20000):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_bullets = 9  # 9颗子弹封顶，每颗10%仓位，最大90%仓位
        self.bullet_size = 0.1
        self.max_hold_days = 20  # V21激进版：持仓不超过20天
        self.max_add_per_action = 3
        
        # 动作空间: 0:持有 1-3:加1-3颗 4-6:减1-3颗 7:全清
        self.action_space = spaces.Discrete(8)
        
        # 扩展观测空间 (11个基础指标 + 6个状态特征 = 17维)
        self.feature_cols = [
            'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
            'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio',
            'boll_width', 'atr', 'price_ma20_ratio'
        ]
        self.state_cols = [
            'position_count', 'floating_pnl_pct', 
            'hold_days', 'avg_cost_price_ratio',
            'max_drawdown_since_entry', 'profit_run_days'
        ]
        
        obs_dim = len(self.feature_cols) + len(self.state_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60
        self.balance = self.initial_balance
        self.hold_positions = []
        self.total_score = 0
        self.done = False
        self.prev_floating_profit = 0
        self.max_trade_profit = 0
        self.entry_max_price = 0  # 建仓后最高价，用于计算回撤
        self.consecutive_profit_days = 0  # 连续盈利天数
        self.last_action_step = -10  # 上次交易步数，防止频繁交易
        self.last_add_price = 0  # 最近一次加仓价格，用于计算加仓后涨幅
        self.consecutive_add_count = 0  # 连续加仓次数，3次后强制减仓
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 技术指标特征
        obs = self.df.loc[self.current_step, self.feature_cols].values
        
        # 状态特征
        current_price = self.df.loc[self.current_step, 'close']
        
        if self.hold_positions:
            num_positions = len(self.hold_positions)
            total_cost = sum(self.initial_balance * self.bullet_size for _ in self.hold_positions)
            total_value = sum((self.initial_balance * self.bullet_size) / p['price'] * current_price for p in self.hold_positions)
            avg_cost_price = sum(p['price'] for p in self.hold_positions) / num_positions
            
            floating_pnl_pct = (total_value - total_cost) / total_cost * 100
            hold_days = self.current_step - self.hold_positions[0]['step']
            avg_cost_price_ratio = (current_price - avg_cost_price) / avg_cost_price * 100
            
            # 入场后最大回撤
            if self.max_trade_profit > 0:
                drawdown_since_entry = (self.max_trade_profit - (total_value - total_cost)) / total_cost * 100
            else:
                drawdown_since_entry = 0
        else:
            num_positions = 0
            floating_pnl_pct = 0
            hold_days = 0
            avg_cost_price_ratio = 0
            drawdown_since_entry = 0
        
        state_features = np.array([
            num_positions / self.max_bullets,  # 归一化仓位
            floating_pnl_pct / 10,  # 浮盈率(缩小范围)
            hold_days / self.max_hold_days,  # 持仓占比
            avg_cost_price_ratio / 10,  # 成本价差率
            drawdown_since_entry / 5,  # 入场后回撤
            self.consecutive_profit_days / 5  # 连续盈利天数
        ], dtype=np.float32)
        
        return np.concatenate([obs.astype(np.float32), state_features])

    def step(self, action):
        reward = 0
        curr_row = self.df.iloc[self.current_step]
        price = curr_row['close']
        
        # 1. 持仓状态评估
        if self.hold_positions:
            num_bullets = len(self.hold_positions)
            total_cost = num_bullets * (self.initial_balance * self.bullet_size)
            total_shares = sum((self.initial_balance * self.bullet_size) / p['price'] for p in self.hold_positions)
            current_value = total_shares * price
            current_pnl = current_value - total_cost
            current_pnl_pct = current_pnl / total_cost * 100
            
            # A. 子弹耗尽惩罚 (V5优化版：鼓励上升趋势加仓，轻度控制满仓)
            usage_ratio = num_bullets / self.max_bullets
            if usage_ratio >= 1.0:
                reward -= 4.0  # 满仓适度惩罚
            elif usage_ratio >= 0.9:
                reward -= 0.8  # 90%以上轻微惩罚
            # 90%以下不惩罚，鼓励积极建仓
            
            # B. 持仓成本 (V21激进版：减少惩罚，让利润奔跑)
            reward -= 0.01 * num_bullets  # 原来是0.02，减少惩罚
            
            # C. 趋势对齐奖励
            if price > curr_row['MA20']:
                reward += 0.15
            
            # D. 利润增长奖励 (基于资金收益率)
            pnl_change = current_pnl_pct - self.prev_floating_profit
            if pnl_change > 0:
                reward += pnl_change * 2.5  # 2.5倍奖励
                self.consecutive_profit_days += 1
                if self.consecutive_profit_days >= 2:
                    reward += 0.3 * self.consecutive_profit_days  # 连续盈利持续加分
            else:
                reward += pnl_change * 0.5  # 亏损时仅轻微扣分
                self.consecutive_profit_days = max(0, self.consecutive_profit_days - 1)
            
            # E. 加仓后大涨额外奖励 (V21激进版：提高奖励门槛)
            if self.last_add_price > 0 and price > self.last_add_price:
                add_gain_pct = (price - self.last_add_price) / self.last_add_price * 100
                if add_gain_pct >= 3.0:  # V21激进版：门槛提高到3%
                    extra_reward = (add_gain_pct - 3.0) * 2.0  # 3%以上每1%奖励2分
                    reward += extra_reward
                    if add_gain_pct >= 6.0:
                        reward += 3.0  # V21激进版：6%以上追加3分
                    if add_gain_pct >= 10.0:
                        reward += 8.0  # V21激进版：10%以上超级奖励8分
                    self.last_add_price = 0  # 奖励后重置
            
            # F. 浮盈回撤惩罚 (最大回撤控制在6%以内)
            self.max_trade_profit = max(self.max_trade_profit, current_pnl)
            if self.max_trade_profit > 0:
                drawdown = (self.max_trade_profit - current_pnl) / total_cost * 100
                if drawdown > 6:  # 最大回撤6%以内
                    reward -= drawdown * 0.25  # 回撤惩罚加重
            
            # G. 持仓天数加权奖励/惩罚 (V21激进版)
            # 目标持仓: 1-5天(最佳) / 6-10天(理想) / 11-15天(正常) / 16-20天(惩罚) / >20天(超时)
            hold_days = self.current_step - self.hold_positions[0]['step']
            if hold_days <= 5:
                reward += 0.4  # V21激进版：1-5天最佳，奖励提高
            elif hold_days <= 10:
                reward += 0.2  # V21激进版：6-10天理想区间
            elif hold_days < 20:
                reward -= (hold_days - 10) * 0.2  # V21激进版：11-20天轻微惩罚
            # 20天以上由超时机制处理
            
            # H. 快速止盈奖励 (5天内盈利止盈额外奖励)
            if hold_days <= 5 and current_pnl > 0:
                reward += 2.0  # 5天内盈利全清额外奖励
            
            # I. 超时强制清仓
            first_entry_step = self.hold_positions[0]['step']
            if (self.current_step - first_entry_step) >= self.max_hold_days:
                reward += self._close_all_positions(price, reason="TIMEOUT")
                action = 0
            
            self.prev_floating_profit = current_pnl
            
            # 动态止损检查 (浮亏超过6%立即止损)
            if current_pnl_pct <= -6:
                reward -= 30  # 止损惩罚
                reward += self._close_all_positions(price, reason="STOP_LOSS")
                self.done = True
                obs = self._get_observation()
                self.total_score += reward
                return obs, float(reward), self.done, False, {}
        else:
            self.prev_floating_profit = 0
            self.max_trade_profit = 0
            self.consecutive_profit_days = 0

        # 2. 动作执行评估
        current_price = price
        
        # G. 频繁交易惩罚 (降低惩罚)
        days_since_last_trade = self.current_step - self.last_action_step
        if action != 0 and days_since_last_trade < 1:
            reward -= 0.3  # 单日频繁交易轻罚
        
        # 加仓动作
        if action in [1, 2, 3]:
            add_bullets = action
            current_count = len(self.hold_positions)
            available = self.max_bullets - current_count
            
            # J. 累计加仓次数限制（每轮持仓最多加仓3次）- 已禁用，回归V13
            add_count_limit_reached = False
            
            # I. 开仓限制：首次建仓最多5颗
            is_new_position = (current_count == 0)
            
            if is_new_position and add_bullets > 5:
                reward -= 2.0  # 开仓超5颗重罚
                add_bullets = 5
            
            if available <= 0:
                reward -= 2.0
            elif add_count_limit_reached:
                reward -= 2.0  # 超过加仓次数限制
            else:
                actual_add = min(add_bullets, available, self.max_add_per_action)
                cost_per = self.initial_balance * self.bullet_size
                total_cost = actual_add * cost_per
                
                if self.balance >= total_cost:
                    # J. 成功加仓，计数+1
                    self.consecutive_add_count += 1
                    
                    # H. 逆势加仓惩罚 (当前亏损时加仓)
                    if self.hold_positions and self.prev_floating_profit < 0:
                        reward -= 0.5 * actual_add
                    
                    self.balance -= total_cost
                    for _ in range(actual_add):
                        self.hold_positions.append({'price': current_price, 'step': self.current_step})
                    
                    # 记录加仓价格，用于后续大涨奖励计算
                    self.last_add_price = current_price
                    self.last_action_step = self.current_step
                    
                    reward += 0.05 * actual_add  # 小额正向鼓励
                    reward -= 0.05 * actual_add  # 手续费
                else:
                    reward -= 1.0

        # 减仓动作
        elif action in [4, 5, 6]:
            reduce_bullets = action - 3
            current_count = len(self.hold_positions)
            
            # J. 重置连续加仓计数
            self.consecutive_add_count = 0
            
            if current_count <= 0:
                reward -= 1.0
            else:
                actual_reduce = min(reduce_bullets, current_count)
                total_reward = 0
                
                for i in range(actual_reduce):
                    bullet = self.hold_positions.pop(0)
                    cost = self.initial_balance * self.bullet_size
                    shares = cost / bullet['price']
                    revenue = shares * current_price
                    pnl_pct = (revenue - cost) / cost * 100
                    
                    self.balance += revenue
                    
                    # 减仓盈亏奖励 (与清仓一致)
                    if pnl_pct >= 3:
                        total_reward += 5
                    elif pnl_pct <= -5:
                        total_reward -= 5
                    elif pnl_pct > 0:
                        total_reward += pnl_pct / 3 * 1
                    else:
                        total_reward += pnl_pct / 5 * 1
                
                reward += total_reward
                reward -= 0.05 * actual_reduce  # 手续费
                self.last_action_step = self.current_step
                
                if not self.hold_positions:
                    self.prev_floating_profit = 0
                    self.max_trade_profit = 0

        # 全清动作
        elif action == 7:
            if self.hold_positions:
                reward += self._close_all_positions(current_price, reason="MANUAL")
                reward -= 0.1
                self.last_action_step = self.current_step
                self.consecutive_add_count = 0  # 重置连续加仓计数

        # 3. 步进
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            if self.hold_positions:
                reward += self._close_all_positions(price, reason="END")
        
        obs = self._get_observation()
        self.total_score += reward
        
        return obs, float(reward), self.done, False, {}

    def _close_all_positions(self, current_price, reason="MANUAL"):
        if not self.hold_positions: return 0
        
        reward = 0
        num_bullets = len(self.hold_positions)
        total_cost = num_bullets * (self.initial_balance * self.bullet_size)
        total_shares = sum((self.initial_balance * self.bullet_size) / p['price'] for p in self.hold_positions)
        revenue = total_shares * current_price
        profit_pct = (revenue - total_cost) / total_cost * 100
        
        if reason == "TIMEOUT":
            # 超时清仓：最大亏损控制在8%以内
            loss_val = abs(profit_pct) if profit_pct < 0 else 0
            if loss_val > 8: reward -= 50   # 亏损>8%重罚
            elif loss_val > 5: reward -= 10  # 亏损5-8%中等惩罚
            elif loss_val > 3: reward -= 5   # 亏损3-5%轻微惩罚
            elif profit_pct < 0: reward -= 2 # 亏损<3%轻微惩罚
            else:
                if profit_pct > 10: reward += 8  # 盈利>10%高奖励
                elif profit_pct >= 5: reward += 5
                else: reward += 2
        else:
            # V21激进版：盈利奖励提高，亏损惩罚降低
            if profit_pct >= 5.0: reward += 8  # V21激进版：原来是3
            elif profit_pct >= 3.0: reward += 5
            elif profit_pct <= -5.0: reward -= 3  # V21激进版：原来是5
            else: reward += profit_pct * 0.8  # V21激进版
            
            if profit_pct > 0 and num_bullets > 1:
                reward += 8  # V21激进版：原来是5
                
            if profit_pct < 0:
                reward -= 3 if abs(profit_pct) > 5 else 1  # V21激进版：减少惩罚
        
        self.balance += revenue
        self.hold_positions = []
        self.prev_floating_profit = 0
        self.max_trade_profit = 0
        return reward
