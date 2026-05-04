import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def get_hist_data(symbol="000001", days=100):
    """获取历史K线数据并调试列名"""
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if not df.empty:
            print("数据列名:", df.columns.tolist())
            return df
    except Exception as e:
        print(f"获取失败: {e}")
    return pd.DataFrame()

if __name__ == "__main__":
    get_hist_data("000001")
