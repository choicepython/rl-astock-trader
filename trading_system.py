import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class TradingSystem:
    def __init__(self, symbol="000001", period="daily", interval="5"):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.hold_position = False
        self.buy_price = 0
        self.log_file = "trading_log.txt"
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []

    def format_symbol(self, symbol):
        if symbol.startswith(('60', '68', '90')):
            return f"sh{symbol}"
        else:
            return f"sz{symbol}"

    def fetch_data(self):
        for _ in range(3):
            try:
                if self.period == "daily":
                    full_symbol = self.format_symbol(self.symbol)
                    df = ak.stock_zh_a_daily(symbol=full_symbol, adjust="qfq")
                else:
                    df = ak.stock_zh_a_hist_min_em(symbol=self.symbol, period=self.interval, adjust="qfq")
                
                if df is not None and not df.empty:
                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'datetime'})
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
            except Exception as e:
                print(f"数据获取失败，正在重试... ({e})")
                time.sleep(2)
        return None

    def calculate_indicators(self, df):
        """计算全套技术指标"""
        # 1. MA
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()

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
        df['j'] = 3 * df['k'] - 2 * df['d']

        # 4. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # 5. WR (Williams %R)
        df['wr'] = (high_9 - df['close']) / (high_9 - low_9) * -100

        # 6. BIAS
        df['bias6'] = (df['close'] - df['close'].rolling(6).mean()) / df['close'].rolling(6).mean() * 100

        # 7. BOLL
        df['boll_mid'] = df['close'].rolling(20).mean()
        df['boll_std'] = df['close'].rolling(20).std()
        df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_low'] = df['boll_mid'] - 2 * df['boll_std']

        # 8. CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(20).mean()
        md_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - ma_tp) / (0.015 * md_tp)

        # 9. 量比 (Volume Ratio)
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma5']

        # 10. 神奇九转 (Magic Nine Turn)
        # 连续9天收盘价高于/低于4天前的收盘价
        df['close_ref4'] = df['close'].shift(4)
        df['up_count'] = (df['close'] > df['close_ref4']).astype(int)
        df['dn_count'] = (df['close'] < df['close_ref4']).astype(int)
        
        # 简单实现连续计数
        def get_consecutive_counts(series):
            counts = []
            cur = 0
            for val in series:
                if val == 1: cur += 1
                else: cur = 0
                counts.append(cur)
            return counts
        
        df['magic_up'] = get_consecutive_counts(df['up_count'])
        df['magic_dn'] = get_consecutive_counts(df['dn_count'])

        return df.fillna(0)

    def prepare_ml_data(self, df):
        """准备机器学习特征和标签"""
        # 特征列
        self.feature_cols = [
            'dif', 'dea', 'macd', 'k', 'd', 'j', 'rsi', 'wr', 'bias6', 
            'boll_up', 'boll_low', 'cci', 'vol_ratio', 'magic_up', 'magic_dn',
            'MA5', 'MA10', 'MA20'
        ]
        
        # 标签：如果未来3个周期内最高价涨幅超过2%，则标记为1（买入机会）
        df['future_pct'] = (df['close'].shift(-3) - df['close']) / df['close']
        df['label'] = (df['future_pct'] > 0.02).astype(int)
        
        # 移除含有 NaN 的行
        data = df.dropna(subset=self.feature_cols + ['label'])
        X = data[self.feature_cols]
        y = data['label']
        
        return X, y

    def train_model(self):
        """训练预测模型：使用数据的前3年"""
        print(f"--- 正在训练模型: {self.symbol} ---")
        df = self.fetch_data()
        if df is None or len(df) < 100:
            print("数据量不足，无法训练模型")
            return
        
        # 筛选前3年的数据进行训练
        first_date = df['datetime'].min()
        train_end_date = first_date + pd.Timedelta(days=3*365)
        train_df = df[df['datetime'] <= train_end_date].copy()
        
        print(f"训练数据范围: {first_date.date()} 至 {train_df['datetime'].max().date()}")
        
        train_df = self.calculate_indicators(train_df)
        X, y = self.prepare_ml_data(train_df)
        
        if X.empty:
            print("训练特征为空，请检查数据量或指标计算")
            return

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用随机森林
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        print("模型训练完成")

    def predict_signal(self, last_row_df):
        """使用模型预测信号"""
        if self.model is None:
            return 0
        
        try:
            features = last_row_df[self.feature_cols]
            features_scaled = self.scaler.transform(features)
            prob = self.model.predict_proba(features_scaled)[0][1] # 获取标记为1的概率
            return prob
        except Exception as e:
            # 防止特征不完整导致预测失败
            return 0

    def check_signals(self, df):
        """结合指标与模型预测生成买卖信号"""
        if len(df) < 20: return "HOLD", 0, None
        
        last = df.iloc[-1:]
        prob = self.predict_signal(last)
        
        # 组合逻辑：模型预测概率 > 0.6 且 指标未超买
        last_val = last.iloc[0]
        
        if not self.hold_position:
            # 买入条件：模型看好 + RSI不超买 + MACD红柱
            if prob > 0.6 and last_val['rsi'] < 70 and last_val['macd'] > 0:
                return "BUY", last_val['close'], last_val['datetime']
        else:
            # 卖出条件：模型不看好 或 RSI超买 或 神奇九转达到9
            if prob < 0.4 or last_val['rsi'] > 85 or last_val['magic_up'] >= 9:
                return "SELL", last_val['close'], last_val['datetime']
        
        return "HOLD", last_val['close'], last_val['datetime']

    def log_trade(self, action, price, dt):
        msg = f"[{dt}] {action} | 价格: {price:.2f} | 股票: {self.symbol}\n"
        print(msg, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg)

    def backtest(self):
        """回测模式：使用最近6个月的数据进行化验"""
        print(f"\n--- 开始模型驱动回测 (近6个月): {self.symbol} ---")
        
        # 1. 确保模型已训练（使用前3年数据）
        if self.model is None:
            self.train_model()
            
        # 2. 获取全量数据并计算指标
        df = self.fetch_data()
        if df is None: return
        df = self.calculate_indicators(df)
        
        # 3. 筛选最近6个月的数据进行回测
        last_date = df['datetime'].max()
        test_start_date = last_date - pd.Timedelta(days=180)
        test_df = df[df['datetime'] >= test_start_date].copy()
        
        if test_df.empty:
            print("回测数据范围不足（近6个月无数据）")
            return
            
        print(f"回测验证范围: {test_df['datetime'].min().date()} 至 {last_date.date()}")
        
        cash = 10000
        initial_cash = cash
        shares = 0
        trade_count = 0
        total_hold_time = 0
        buy_time = None
        
        # 遍历回测区间
        for i in range(len(test_df)):
            # 获取当前点之前的完整上下文以准确计算信号
            current_dt = test_df.iloc[i]['datetime']
            full_sub_df = df[df['datetime'] <= current_dt]
            
            signal, price, dt = self.check_signals(full_sub_df)
            
            if signal == "BUY" and not self.hold_position:
                self.hold_position = True
                self.buy_price = price
                buy_time = dt
                shares = cash // price
                cash -= shares * price
                print(f"[{dt}] 买入: {price:.2f}")
            elif signal == "SELL" and self.hold_position:
                # 计算持仓时间
                duration = (dt - buy_time).days if self.period == "daily" else (dt - buy_time).total_seconds() / 60
                total_hold_time += duration
                trade_count += 1
                
                cash += shares * price
                profit_pct = (price - self.buy_price) / self.buy_price * 100
                unit = "天" if self.period == "daily" else "分钟"
                print(f"[{dt}] 卖出: {price:.2f}, 收益: {profit_pct:.2f}%, 持仓: {duration:.1f}{unit}")
                self.hold_position = False
                shares = 0
                buy_time = None
        
        final_value = cash + (shares * test_df.iloc[-1]['close'] if shares > 0 else 0)
        total_return_pct = (final_value - initial_cash) / initial_cash * 100
        avg_hold_time = total_hold_time / trade_count if trade_count > 0 else 0
        unit = "天" if self.period == "daily" else "分钟"
        
        print("\n" + "="*40)
        print(f"回测总结 ({self.symbol})")
        print(f"最终资产: {final_value:.2f}")
        print(f"整体收益率: {total_return_pct:.2f}%")
        print(f"总交易次数: {trade_count}")
        print(f"平均持仓时间: {avg_hold_time:.2f} {unit}")
        print("="*40)

    def monitor(self, duration_minutes=60):
        print(f"\n开始模型实盘监控: {self.symbol}")
        if self.model is None:
            self.train_model()
            
        end_time = time.time() + duration_minutes * 60
        while time.time() < end_time:
            df = self.fetch_data()
            if df is not None and len(df) > 50:
                df = self.calculate_indicators(df)
                signal, price, dt = self.check_signals(df)
                prob = self.predict_signal(df.iloc[-1:])
                
                status_msg = f"\r时间: {datetime.now().strftime('%H:%M:%S')} | 现价: {price:.2f} | 模型看多率: {prob:.2%} | 状态: {'持仓' if self.hold_position else '空仓'}"
                print(status_msg, end="")
                
                if signal == "BUY":
                    self.hold_position = True
                    self.buy_price = price
                    self.log_trade("【AI买入信号】", price, dt)
                elif signal == "SELL":
                    profit = (price - self.buy_price) / self.buy_price * 100
                    self.hold_position = False
                    self.log_trade(f"【AI卖出信号】(预期收益: {profit:.2f}%)", price, dt)
            
            time.sleep(60 if self.period == "min" else 300)

if __name__ == "__main__":
    # 使用 002079 进行模型回测与实盘预测
    system = TradingSystem(symbol="002079", period="daily")
    
    # 1. 训练模型并回测
    system.backtest()
    
    # 2. 如果需要实时预测，请取消注释
    # system.monitor(duration_minutes=10)
