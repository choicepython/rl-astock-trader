import akshare as ak
import pandas as pd

def format_symbol(symbol):
    return f"sh{symbol}" if symbol.startswith(('60', '68', '90')) else f"sz{symbol}"

# 获取数据
df = ak.stock_zh_a_daily(symbol=format_symbol('002156'), adjust='qfq')
print(f"总数据行数: {len(df)}")
print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
print(f"2026年数据行数: {len(df[df['date'] >= '2026-01-01'])}")
print(f"\n最近10条数据:")
print(df.tail(10))
