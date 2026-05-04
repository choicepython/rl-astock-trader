"""
数据获取与指标计算模块
"""
import akshare as ak
import pandas as pd
import numpy as np

def format_stock_code(code: str) -> str:
    """格式化股票代码"""
    if code.startswith(('60', '68', '90')):
        return f'sh' + code
    return f'sz' + code

def fetch_stock_data(stock_code: str, days: int = 120) -> pd.DataFrame:
    """
    获取股票历史数据
    
    Args:
        stock_code: 股票代码，如002156
        days: 获取天数
    
    Returns:
        DataFrame with OHLCV数据
    """
    for _ in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=format_stock_code(stock_code))
            if df is not None and not df.empty:
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'datetime'})
                df['datetime'] = pd.to_datetime(df['datetime'])
                cols_map = {'开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}
                df = df.rename(columns=cols_map)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.tail(days).reset_index(drop=True)
        except Exception as e:
            print(f"数据获取重试: {e}")
            continue
    return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标 (14个)
    
    Returns:
        DataFrame with technical indicators
    """
    df = df.copy()
    
    # 1. 均线系统
    for period in [5, 10, 20, 60]:
        df[f'MA{period}'] = df['close'].rolling(period).mean()
    
    # 2. MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = exp12 - exp26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # 3. KD随机指标
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
    
    # 5. BOLL带宽
    df['boll_mid'] = df['close'].rolling(20).mean()
    df['boll_std'] = df['close'].rolling(20).std()
    boll_upper = df['boll_mid'] + 2 * df['boll_std']
    boll_lower = df['boll_mid'] - 2 * df['boll_std']
    df['boll_width'] = (boll_upper - boll_lower) / df['boll_mid']
    
    # 6. ATR波动率
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean() / df['close']
    
    # 7. 价格相对均线位置
    df['price_ma20_ratio'] = (df['close'] - df['MA20']) / df['MA20'] * 100
    
    # 8. 量比
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    
    return df.fillna(0)
