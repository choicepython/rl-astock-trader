import akshare as ak
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 获取股票数据 (使用akshare获取平安银行 000001 的历史数据)
def get_stock_data(symbol="000001", start_date="20230101", end_date="20231231"):
    print(f"正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    
    # 转换列名以符合 backtrader 的要求
    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

# 2. 定义量化策略 (双均线策略)
class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 5),   # 短期均线周期
        ('slow_period', 20),  # 长期均线周期
    )

    def __init__(self):
        # 计算短期和长期移动平均线
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # 交叉信号：1为金叉（买入），-1为死叉（卖出）
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        # 如果当前没有持仓，且发生金叉，买入
        if not self.position:
            if self.crossover > 0:
                self.buy()
                print(f"{self.datetime.date(0)}: 买入信号，价格: {self.data.close[0]:.2f}")
        
        # 如果当前有持仓，且发生死叉，卖出
        elif self.crossover < 0:
            self.close()
            print(f"{self.datetime.date(0)}: 卖出信号，价格: {self.data.close[0]:.2f}")

# 3. 主函数：运行回测
def run_backtest():
    # 初始化回测引擎
    cerebro = bt.Cerebro()

    # 加载数据
    data_df = get_stock_data()
    data = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(SmaCrossStrategy)

    # 设置初始资金
    cerebro.broker.setcash(10000.0)
    
    # 设置佣金 (例如 0.1%)
    cerebro.broker.setcommission(commission=0.001)

    # 打印初始状态
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")

    # 运行回测
    cerebro.run()

    # 打印最终状态
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    
    # 绘制回测图
    # 在某些环境下绘图可能需要 backend 设置，这里使用默认设置
    try:
        cerebro.plot(style='candlestick')
    except Exception as e:
        print(f"绘图出错 (可能是环境限制): {e}")

if __name__ == "__main__":
    run_backtest()
