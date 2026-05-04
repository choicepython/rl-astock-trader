import akshare as ak
import pandas as pd
import random
import time
from tqdm import tqdm
import os

def collect_data(sample_size=3000, start_date="20200101"):
    print("正在获取全量股票列表...")
    try:
        # 尝试使用新浪接口获取列表，通常更稳定
        stock_list_df = ak.stock_zh_a_spot()
        # 新浪接口返回的代码是 sh000001 格式，提取纯数字
        all_symbols = [s[2:] for s in stock_list_df['代码'].tolist()]
    except:
        # 备选接口
        stock_list_df = ak.stock_zh_a_spot_em()
        all_symbols = stock_list_df['代码'].tolist()
    
    # 随机抽取
    if len(all_symbols) > sample_size:
        selected_symbols = random.sample(all_symbols, sample_size)
    else:
        selected_symbols = all_symbols
        
    print(f"已选取 {len(selected_symbols)} 支股票，准备获取历史数据...")
    
    all_data = []
    output_file = "data.csv"
    
    # 如果文件已存在，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
        
    count = 0
    for symbol in tqdm(selected_symbols):
        try:
            # 优先使用新浪接口获取历史数据，更稳定
            # 注意：sina 接口 symbol 需要 sh/sz 前缀
            def format_sina_symbol(s):
                return f"sh{s}" if s.startswith(('60', '68', '90')) else f"sz{s}"
            
            df = ak.stock_zh_a_daily(symbol=format_sina_symbol(symbol), adjust="qfq")
            
            if df is not None and not df.empty:
                # 统一列名以匹配后续处理逻辑
                # sina 返回: date, open, high, low, close, volume
                df = df.rename(columns={
                    'date': '日期', 'open': '开盘', 'high': '最高', 
                    'low': '最低', 'close': '收盘', 'volume': '成交量'
                })
                # 转换日期格式，过滤 2020 年以后的数据
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= start_date]
                
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
                    count += 1
                
            # 每 100 支股票合并保存一次，防止内存溢出或崩溃
            if len(all_data) >= 100:
                temp_df = pd.concat(all_data)
                temp_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                all_data = []
                
            # 适当休眠，避免被封IP
            time.sleep(0.1)
        except Exception as e:
            # print(f"\n获取 {symbol} 失败: {e}")
            continue
            
    # 保存剩余数据
    if all_data:
        temp_df = pd.concat(all_data)
        temp_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        
    print(f"\n数据采集完成，共获取 {count} 支股票数据，保存至 {output_file}")

if __name__ == "__main__":
    # 注意：下载3000支股票数据量巨大且耗时较长
    # 为了演示，我们先尝试一个较小的样本，用户可根据需要自行修改为 3000
    collect_data(sample_size=3000)
