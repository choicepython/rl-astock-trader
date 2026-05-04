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
        self.max_hold_days = 15  # V23优化：缩短持仓周期，增加交易频次
        self.max_add_per_action = 3
        
        # 动作空间: 0:持有 1-3:加1-3颗 4-6:减1-3颗 7:全清
        self.action_space = spaces.Discrete(8)
        
        # 扩展观测空间 (14个基础指标 + 8个状态特征 = 22维)
        self.feature_cols = [
            'dif', 'dea', 'macd', 'k', 'd', 'rsi', 
            'MA5', 'MA10', 'MA20', 'MA60', 'vol_ratio',
            'boll_width', 'atr', 'price_ma20_ratio'
        ]
        self.state_cols = [
            'position_count', 'floating_pnl_pct', 
            'hold_days', 'avg_cost_price_ratio',
            'max_drawdown_since_entry', 'profit_run_days',
            'trailing_stop_distance', 'atr_multiplier'
        ]
        
        obs_dim = len(self.feature_cols) + len(self.state_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动态止损参数
        self.atr_stop_multiplier = 2.0  # ATR止损倍数
        self.trailing_stop_enabled = True  # 启用追踪止损
        self.min_trailing_stop = 5.0  # 优化：最小追踪止损百分比从3%提高到5%
        self.stop_loss_cooldown = 0  # 优化：止损冷静期
        
        # 胜率统计
        self.win_count = 0
        self.total_trades = 0
        
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
        self.last_action_step = -5  # 上次交易步数，降低频繁交易限制
        self.last_add_price = 0  # 最近一次加仓价格
        self.consecutive_add_count = 0  # 连续加仓次数
        
        # 追踪止损相关
        self.trailing_stop_price = 0  # 当前追踪止损价
        self.trailing_stop_initialized = False
        
        # 胜率统计重置
        self.win_count = 0
        self.total_trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 技术指标特征
        obs = self.df.loc[self.current_step, self.feature_cols].values
        
        # 状态特征
        current_price = self.df.loc[self.current_step, 'close']
        atr_value = self.df.loc[self.current_step, 'atr'] if 'atr' in self.df.columns else 0.01
        
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
            
            # 追踪止损距离
            if self.trailing_stop_enabled and self.trailing_stop_price > 0:
                trailing_stop_distance = (current_price - self.trailing_stop_price) / self.trailing_stop_price * 100
            else:
                trailing_stop_distance = 0
        else:
            num_positions = 0
            floating_pnl_pct = 0
            hold_days = 0
            avg_cost_price_ratio = 0
            drawdown_since_entry = 0
            trailing_stop_distance = 0
        
        state_features = np.array([
            num_positions / self.max_bullets,  # 归一化仓位
            floating_pnl_pct / 10,  # 浮盈率(缩小范围)
            hold_days / self.max_hold_days,  # 持仓占比
            avg_cost_price_ratio / 10,  # 成本价差率
            drawdown_since_entry / 5,  # 入场后回撤
            self.consecutive_profit_days / 5,  # 连续盈利天数
            trailing_stop_distance / 5,  # 追踪止损距离
            atr_value * 100  # ATR值放大
        ], dtype=np.float32)
        
        return np.concatenate([obs.astype(np.float32), state_features])

    def step(self, action):
        reward = 0
        curr_row = self.df.iloc[self.current_step]
        price = curr_row['close']
        atr_value = curr_row['atr'] if 'atr' in curr_row else 0.01
        
        # 1. 持仓状态评估
        if self.hold_positions:
            num_bullets = len(self.hold_positions)
            total_cost = num_bullets * (self.initial_balance * self.bullet_size)
            total_shares = sum((self.initial_balance * self.bullet_size) / p['price'] for p in self.hold_positions)
            current_value = total_shares * price
            current_pnl = current_value - total_cost
            current_pnl_pct = current_pnl / total_cost * 100
            
            # ==================== 优化1: 奖励函数改进 ====================
            # A. 降低满仓惩罚，鼓励积极建仓
            usage_ratio = num_bullets / self.max_bullets
            if usage_ratio >= 1.0:
                reward -= 0.5  # 优化：从1.5降到0.5
            elif usage_ratio >= 0.9:
                reward -= 0.2  # 优化：从0.3降到0.2
            
            # B. 降低持仓成本惩罚，鼓励持有
            reward -= 0.002 * num_bullets  # 优化：从0.005降到0.002
            
            # C. 增强趋势对齐奖励
            if price > curr_row['MA20']:
                reward += 0.2  # V23: 从0.15提高到0.2
                # 多头排列额外奖励
                if curr_row['MA5'] > curr_row['MA10'] and curr_row['MA10'] > curr_row['MA20']:
                    reward += 0.15
            
            # D. 利润增长奖励 - 增强版
            pnl_change = current_pnl_pct - self.prev_floating_profit
            if pnl_change > 0:
                reward += pnl_change * 4.0  # 优化：从3.0提高到4.0
                self.consecutive_profit_days += 1
                if self.consecutive_profit_days >= 2:
                    reward += 0.5 * self.consecutive_profit_days  # 优化：从0.4提高到0.5
            else:
                reward += pnl_change * 0.2  # 优化：从0.3降到0.2
                self.consecutive_profit_days = max(0, self.consecutive_profit_days - 1)
            
            # E. 小额盈利即时奖励 (V23新增：提高胜率导向)
            if current_pnl_pct > 1.0 and current_pnl_pct < 5.0:
                reward += 0.5  # 小额盈利鼓励止盈
            
            # F. 加仓后上涨奖励 (降低门槛)
            if self.last_add_price > 0 and price > self.last_add_price:
                add_gain_pct = (price - self.last_add_price) / self.last_add_price * 100
                if add_gain_pct >= 2.0:  # V23: 门槛从3%降到2%
                    extra_reward = (add_gain_pct - 2.0) * 2.5
                    reward += extra_reward
                    if add_gain_pct >= 5.0:
                        reward += 4.0
                    if add_gain_pct >= 8.0:
                        reward += 10.0
                    self.last_add_price = 0
            
            # G. 浮盈回撤惩罚 - 非线性加重 (V23优化)
            self.max_trade_profit = max(self.max_trade_profit, current_pnl)
            if self.max_trade_profit > 0:
                drawdown = (self.max_trade_profit - current_pnl) / total_cost * 100
                if drawdown > 5:
                    reward -= drawdown ** 1.2 * 0.15  # 非线性惩罚
            
            # H. 持仓天数优化 (V23：缩短最佳持仓周期)
            hold_days = self.current_step - self.hold_positions[0]['step']
            if hold_days <= 4:
                reward += 0.5  # V23: 1-4天最佳，奖励提高到0.5
            elif hold_days <= 8:
                reward += 0.25  # V23: 5-8天理想
            elif hold_days <= 12:
                reward += 0.1  # V23: 9-12天正常
            elif hold_days < self.max_hold_days:
                reward -= (hold_days - 12) * 0.15  # V23: 13-15天轻微惩罚
            
            # I. 快速止盈奖励增强
            if hold_days <= 4 and current_pnl > 0:
                reward += 3.0  # V23: 从2.0提高到3.0
            
            # J. 更新追踪止损 (V23新增)
            self._update_trailing_stop(price, total_cost, num_bullets)
            
            # K. 超时强制清仓
            first_entry_step = self.hold_positions[0]['step']
            if (self.current_step - first_entry_step) >= self.max_hold_days:
                reward += self._close_all_positions(price, reason="TIMEOUT")
                action = 0
            
            self.prev_floating_profit = current_pnl
            
            # ==================== 优化2: 动态止损机制 ====================
            # 动态ATR止损 + 追踪止损
            stop_loss_triggered = False
            
            # ATR止损检查
            atr_stop_level = avg_cost_price = sum(p['price'] for p in self.hold_positions) / num_bullets
            atr_stop_price = avg_cost_price * (1 - self.atr_stop_multiplier * atr_value)
            
            # 追踪止损检查
            if self.trailing_stop_enabled and self.trailing_stop_price > 0:
                if price <= self.trailing_stop_price:
                    stop_loss_triggered = True
                    stop_reason = "TRAILING_STOP"
            elif current_pnl_pct <= -5:  # V23: 止损线从6%降到5%
                stop_loss_triggered = True
                stop_reason = "FIXED_STOP"
            
            if stop_loss_triggered:
                reward -= 15  # V23: 止损惩罚大幅降低，从30降到15
                reward += self._close_all_positions(price, reason=stop_reason)
                self.stop_loss_cooldown = 3  # 优化：止损后设置3天冷静期
                self.done = True
                obs = self._get_observation()
                self.total_score += reward
                return obs, float(reward), self.done, False, {}
        else:
            self.prev_floating_profit = 0
            self.max_trade_profit = 0
            self.consecutive_profit_days = 0
            self.trailing_stop_price = 0
            self.trailing_stop_initialized = False

        # 2. 动作执行评估
        current_price = price
        
        # 降低频繁交易惩罚 (V23优化)
        days_since_last_trade = self.current_step - self.last_action_step
        if action != 0 and days_since_last_trade < 1:
            reward -= 0.1  # V23: 从0.3降到0.1
        
        # 加仓动作
        if action in [1, 2, 3]:
            add_bullets = action
            current_count = len(self.hold_positions)
            available = self.max_bullets - current_count
            
            # 优化：止损冷静期检查
            if self.stop_loss_cooldown > 0:
                self.stop_loss_cooldown -= 1
                reward -= 1.0  # 冷静期内尝试加仓惩罚
                action = 0  # 改为持有
            elif available <= 0:
                reward -= 1.0  # V23: 惩罚降低，从2.0降到1.0
            else:
                # 开仓限制放宽 (V23优化：从5颗增加到7颗)
                is_new_position = (current_count == 0)
                max_open_bullets = 7  # V23: 从5增加到7
                
                if is_new_position and add_bullets > max_open_bullets:
                    reward -= 1.0  # V23: 惩罚降低，从2.0降到1.0
                    add_bullets = max_open_bullets
                
                actual_add = min(add_bullets, available, self.max_add_per_action)
                cost_per = self.initial_balance * self.bullet_size
                total_cost = actual_add * cost_per
                
                if self.balance >= total_cost:
                    self.consecutive_add_count += 1
                    
                    # 逆势加仓惩罚降低
                    if self.hold_positions and self.prev_floating_profit < 0:
                        reward -= 0.3 * actual_add  # V23: 从0.5降到0.3
                    
                    self.balance -= total_cost
                    for _ in range(actual_add):
                        self.hold_positions.append({'price': current_price, 'step': self.current_step})
                    
                    self.last_add_price = current_price
                    self.last_action_step = self.current_step
                    
                    # 增加加仓奖励
                    reward += 0.1 * actual_add  # V23: 从0.05提高到0.1
                    reward -= 0.03 * actual_add  # V23: 手续费降低，从0.05降到0.03
                    
                    # 初始化追踪止损
                    if self.trailing_stop_enabled and not self.trailing_stop_initialized:
                        self.trailing_stop_price = current_price * (1 - self.min_trailing_stop / 100)
                        self.trailing_stop_initialized = True
                else:
                    reward -= 0.5  # V23: 惩罚降低，从1.0降到0.5

        # 减仓动作
        elif action in [4, 5, 6]:
            reduce_bullets = action - 3
            current_count = len(self.hold_positions)
            
            self.consecutive_add_count = 0
            
            if current_count <= 0:
                reward -= 0.5  # V23: 惩罚降低，从1.0降到0.5
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
                    
                    # 优化减仓奖励结构 (V23)
                    if pnl_pct >= 2:
                        total_reward += 6  # V23: 从5提高到6
                    elif pnl_pct >= 1:
                        total_reward += 3  # V23: 新增1-2%盈利奖励
                    elif pnl_pct <= -3:
                        total_reward -= 3  # V23: 从-5降到-3
                    elif pnl_pct > 0:
                        total_reward += pnl_pct * 1.5  # V23: 从1.0提高到1.5
                    else:
                        total_reward += pnl_pct * 0.5  # V23: 从1.0降到0.5
                
                reward += total_reward
                reward -= 0.03 * actual_reduce  # V23: 手续费降低
                self.last_action_step = self.current_step
                
                if not self.hold_positions:
                    self.prev_floating_profit = 0
                    self.max_trade_profit = 0
                    self.trailing_stop_price = 0
                    self.trailing_stop_initialized = False

        # 全清动作
        elif action == 7:
            if self.hold_positions:
                reward += self._close_all_positions(current_price, reason="MANUAL")
                reward -= 0.05  # V23: 手续费降低，从0.1降到0.05
                self.last_action_step = self.current_step
                self.consecutive_add_count = 0

        # 3. 步进
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            if self.hold_positions:
                reward += self._close_all_positions(price, reason="END")
        
        obs = self._get_observation()
        self.total_score += reward
        
        return obs, float(reward), self.done, False, {}

    def _update_trailing_stop(self, current_price, total_cost, num_bullets):
        """更新追踪止损价格"""
        if not self.trailing_stop_enabled or num_bullets == 0:
            return
        
        avg_cost_price = sum(p['price'] for p in self.hold_positions) / num_bullets
        current_pnl_pct = (current_price - avg_cost_price) / avg_cost_price * 100
        
        # 只有在盈利超过最小追踪止损时才启用追踪
        if current_pnl_pct >= self.min_trailing_stop:
            # 新的止损价 = 当前价 - min_trailing_stop%
            new_stop_price = current_price * (1 - self.min_trailing_stop / 100)
            
            # 只向上移动止损
            if new_stop_price > self.trailing_stop_price:
                self.trailing_stop_price = new_stop_price

    def _close_all_positions(self, current_price, reason="MANUAL"):
        if not self.hold_positions: return 0
        
        reward = 0
        num_bullets = len(self.hold_positions)
        total_cost = num_bullets * (self.initial_balance * self.bullet_size)
        total_shares = sum((self.initial_balance * self.bullet_size) / p['price'] for p in self.hold_positions)
        revenue = total_shares * current_price
        profit_pct = (revenue - total_cost) / total_cost * 100
        
        # 更新胜率统计
        self.total_trades += 1
        if profit_pct > 0:
            self.win_count += 1
        
        if reason == "TIMEOUT":
            # 超时清仓：优化奖励结构
            loss_val = abs(profit_pct) if profit_pct < 0 else 0
            if loss_val > 6:
                reward -= 30  # V23: 从50降到30
            elif loss_val > 4:
                reward -= 8  # V23: 从10降到8
            elif loss_val > 2:
                reward -= 3  # V23: 从5降到3
            elif profit_pct < 0:
                reward -= 1  # V23: 从2降到1
            else:
                if profit_pct > 8:
                    reward += 10  # V23: 从8提高到10
                elif profit_pct >= 4:
                    reward += 6  # V23: 从5提高到6
                elif profit_pct >= 2:
                    reward += 3  # V23: 从2提高到3
                else:
                    reward += 1
        else:
            # V23优化版：增强盈利奖励，降低亏损惩罚
            if profit_pct >= 5.0:
                reward += 10  # V23: 从8提高到10
            elif profit_pct >= 3.0:
                reward += 6  # V23: 从5提高到6
            elif profit_pct >= 1.0:
                reward += 3  # V23: 新增1-3%盈利奖励
            elif profit_pct <= -4.0:
                reward -= 2  # V23: 从-3降到-2
            else:
                reward += profit_pct * 1.0  # V23: 从0.8提高到1.0
            
            # 多仓位盈利额外奖励
            if profit_pct > 0:
                reward += 2 * num_bullets  # V23: 按仓位数量奖励
            
            if profit_pct < 0:
                reward -= 2 if abs(profit_pct) > 4 else 0.5  # V23: 降低惩罚
        
        self.balance += revenue
        self.hold_positions = []
        self.prev_floating_profit = 0
        self.max_trade_profit = 0
        self.trailing_stop_price = 0
        self.trailing_stop_initialized = False
        
        return reward
