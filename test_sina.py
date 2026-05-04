import akshare as ak
import pandas as pd

def test_sina():
    try:
        print("尝试从新浪获取实时数据...")
        df = ak.stock_zh_a_spot()
        if not df.empty:
            print("新浪实时数据获取成功！")
            print(df.head())
            return
    except Exception as e:
        print(f"新浪实时获取失败: {e}")

    try:
        print("\n尝试从新浪获取历史数据...")
        df = ak.stock_zh_a_daily(symbol="sh000001")
        if not df.empty:
            print("新浪历史数据获取成功！")
            print(df.head())
    except Exception as e:
        print(f"新浪历史获取失败: {e}")

if __name__ == "__main__":
    test_sina()
