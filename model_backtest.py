import akshare as ak
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def calculate_indicators(df):
    df = df.sort_values('日期')
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['dif'] = exp1 - exp2
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['wr'] = (high_9 - df['收盘']) / (high_9 - low_9) * -100
    df['bias6'] = (df['收盘'] - df['收盘'].rolling(6).mean()) / df['收盘'].rolling(6).mean() * 100
    df['boll_mid'] = df['收盘'].rolling(20).mean()
    df['boll_std'] = df['收盘'].rolling(20).std()
    df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_low'] = df['boll_mid'] - 2 * df['boll_std']
    tp = (df['最高'] + df['最低'] + df['收盘']) / 3
    ma_tp = tp.rolling(20).mean()
    md_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - ma_tp) / (0.015 * md_tp)
    df['vol_ma5'] = df['成交量'].rolling(5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    df['close_ref4'] = df['收盘'].shift(4)
    df['up_count'] = (df['收盘'] > df['close_ref4']).astype(int)
    df['dn_count'] = (df['收盘'] < df['close_ref4']).astype(int)
    def get_consecutive_counts(series):
        counts = []; cur = 0
        for val in series:
            if val == 1: cur += 1
            else: cur = 0
            counts.append(cur)
        return counts
    df['magic_up'] = get_consecutive_counts(df['up_count'])
    df['magic_dn'] = get_consecutive_counts(df['dn_count'])
    return df

def run_backtest(symbol="002079"):
    print(f"\n--- 开始指定股票回测 (2026年至今): {symbol} ---")
    
    # 加载模型
    try:
        model_data = joblib.load('model_weights.joblib')
        clf = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
    except:
        print("错误：未找到模型权重文件，请先运行 train_stacking_model.py")
        return

    # 获取数据（使用新浪接口更稳定）
    try:
        def format_sina_symbol(s):
            return f"sh{s}" if s.startswith(('60', '68', '90')) else f"sz{s}"
        df = ak.stock_zh_a_daily(symbol=format_sina_symbol(symbol), adjust="qfq")
        if df is not None and not df.empty:
            df = df.rename(columns={
                'date': '日期', 'open': '开盘', 'high': '最高', 
                'low': '最低', 'close': '收盘', 'volume': '成交量'
            })
            df['日期'] = pd.to_datetime(df['日期'])
    except Exception as e:
        print(f"获取数据失败: {e}")
        return
    df = calculate_indicators(df)
    df = df.fillna(0)
    
    # 筛选 2026 年以后的数据进行回测
    test_df = df[df['日期'] >= '2026-01-01'].copy()
    if test_df.empty:
        print("2026年至今无数据")
        return
    
    print(f"回测区间: {test_df['日期'].min().date()} 至 {test_df['日期'].max().date()}")
    
    cash = 10000
    initial_cash = cash
    shares = 0
    hold_position = False
    buy_price = 0
    buy_time = None
    trade_count = 0
    total_hold_time = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        dt = row['日期']
        price = row['收盘']
        
        # 准备特征进行预测
        features = row[feature_cols].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prob = clf.predict_proba(features_scaled)[0][1]
        
        # 策略逻辑
        if not hold_position:
            # 买入：AI预测概率 > 0.6 且 RSI < 70
            if prob > 0.6 and row['rsi'] < 70:
                hold_position = True
                buy_price = price
                buy_time = dt
                shares = cash // price
                cash -= shares * price
                print(f"[{dt.date()}] 买入信号 (AI概率: {prob:.2%}), 价格: {price:.2f}")
        else:
            # 卖出：AI预测概率 < 0.4 或 RSI > 85 或 神奇九转达到9
            if prob < 0.4 or row['rsi'] > 85 or row['magic_up'] >= 9:
                trade_count += 1
                cash += shares * price
                profit_pct = (price - buy_price) / buy_price * 100
                duration = (dt - buy_time).days
                total_hold_time += duration
                print(f"[{dt.date()}] 卖出信号 (AI概率: {prob:.2%}), 价格: {price:.2f}, 收益: {profit_pct:.2f}%, 持仓: {duration}天")
                hold_position = False
                shares = 0
                
    final_value = cash + (shares * test_df.iloc[-1]['收盘'] if shares > 0 else 0)
    total_return = (final_value - initial_cash) / initial_cash * 100
    avg_hold_time = total_hold_time / trade_count if trade_count > 0 else 0
    
    print("\n" + "="*40)
    print(f"预训练模型回测总结 ({symbol})")
    print(f"最终资产: {final_value:.2f}")
    print(f"累计收益率: {total_return:.2f}%")
    print(f"总交易次数: {trade_count}")
    print(f"平均持仓时间: {avg_hold_time:.2f} 天")
    print("="*40)

if __name__ == "__main__":
    run_backtest("002079")
