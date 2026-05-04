import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import time

def format_symbol(symbol):
    """将数字代码转换为新浪需要的格式 (sh000001, sz000001)"""
    if symbol.startswith(('60', '68', '90')):
        return f"sh{symbol}"
    else:
        return f"sz{symbol}"

def get_realtime_data_sina(symbol="000001"):
    """使用新浪接口获取实时数据"""
    try:
        df = ak.stock_zh_a_spot()
        full_symbol = format_symbol(symbol)
        stock_data = df[df['代码'] == full_symbol]
        if not stock_data.empty:
            return {
                "name": stock_data['名称'].values[0],
                "price": float(stock_data['最新价'].values[0]),
                "change": float(stock_data['涨跌幅'].values[0]),
                "volume": float(stock_data['成交量'].values[0]),
                "turnover": float(stock_data['成交额'].values[0])
            }
    except Exception as e:
        print(f"实时数据获取失败: {e}")
    return None

def get_hist_data_sina(symbol="000001"):
    """使用新浪接口获取历史数据"""
    try:
        full_symbol = format_symbol(symbol)
        df = ak.stock_zh_a_daily(symbol=full_symbol, adjust="qfq")
        if not df.empty:
            # 新浪返回的列名通常是 date, open, high, low, close, volume
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        print(f"历史数据获取失败: {e}")
    return pd.DataFrame()

def calculate_indicators(df):
    """计算技术指标"""
    if df.empty: return df
    
    # MA
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # KDJ
    low_list = df['low'].rolling(9, min_periods=9).min()
    high_list = df['high'].rolling(9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # BOLL
    df['BOLL_MID'] = df['close'].rolling(window=20).mean()
    df['BOLL_STD'] = df['close'].rolling(window=20).std()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
    df['BOLL_LOW'] = df['BOLL_MID'] - 2 * df['BOLL_STD']

    # WR
    df['WR'] = (high_list - df['close']) / (high_list - low_list) * -100

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md_tp = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * md_tp)

    # BIAS
    df['BIAS6'] = (df['close'] - df['close'].rolling(6).mean()) / df['close'].rolling(6).mean() * 100
    df['BIAS12'] = (df['close'] - df['close'].rolling(12).mean()) / df['close'].rolling(12).mean() * 100

    return df

def main(symbol="000001"):
    print(f"--- 正在分析股票 (新浪接口): {symbol} ---")
    
    # 实时数据
    realtime = get_realtime_data_sina(symbol)
    if realtime:
        print(f"名称: {realtime['name']} | 最新价: {realtime['price']} | 涨跌幅: {realtime['change']}%")
    
    # 历史数据与指标
    df = get_hist_data_sina(symbol)
    if df.empty:
        print("无法获取数据，请检查网络或代码。")
        return
        
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    print("\n--- 最新技术指标 ---")
    print(f"MA: MA5={last['MA5']:.2f}, MA10={last['MA10']:.2f}, MA20={last['MA20']:.2f}")
    print(f"MACD: DIF={last['DIF']:.2f}, DEA={last['DEA']:.2f}, MACD={last['MACD']:.2f}")
    print(f"KDJ: K={last['K']:.2f}, D={last['D']:.2f}, J={last['J']:.2f}")
    print(f"RSI: {last['RSI']:.2f}")
    print(f"BOLL: 上轨={last['BOLL_UP']:.2f}, 中轨={last['BOLL_MID']:.2f}, 下轨={last['BOLL_LOW']:.2f}")
    print(f"WR: {last['WR']:.2f}")
    print(f"CCI: {last['CCI']:.2f}")
    print(f"BIAS: BIAS6={last['BIAS6']:.2f}%, BIAS12={last['BIAS12']:.2f}%")

if __name__ == "__main__":
    main("002079")
