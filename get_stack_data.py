import akshare as ak
import pandas as pd
from prettytable import PrettyTable
import warnings

# 精准忽略无关警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class StockDataFetcher:
    """股票实时数据+技术指标获取器（修复版，支持分钟/日线）"""
    def __init__(self, stock_code: str, period: str = "1", adjust: str = "qfq"):
        """
        初始化参数
        :param stock_code: 股票代码（支持带前缀：sh600000 / sz000001）
        :param period: 时间周期：1/5/15/30/60(分钟) / daily(日线)
        :param adjust: 复权方式：qfq(前复权)/hfq(后复权)/""(不复权)
        """
        # 自动拆分 前缀+纯代码（核心修复）
        self.code_prefix = stock_code[:2]  # sh/sz
        self.code_plain = stock_code[2:]   # 纯数字代码
        self.stock_code = stock_code       # 完整代码
        self.period = period
        self.adjust = adjust
        self.stock_name = ""
        self.realtime_data = None  # 缓存实时数据，避免重复请求

    def _format_number(self, num):
        """格式化大数字：千分位+保留2位小数"""
        if isinstance(num, (int, float)):
            return f"{num:,.2f}"
        return num

    def get_realtime_price(self):
        """获取实时股票价格（东方财富，延迟15秒）"""
        try:
            realtime_df = ak.stock_zh_a_spot_em()
            stock_data = realtime_df[realtime_df["代码"] == self.stock_code]

            if stock_data.empty:
                raise ValueError(f"未找到股票：{self.stock_code}")

            # 缓存数据
            self.stock_name = stock_data["名称"].iloc[0]
            self.realtime_data = {
                "股票名称": self.stock_name,
                "最新价": self._format_number(stock_data["最新价"].iloc[0]),
                "涨跌幅(%)": self._format_number(stock_data["涨跌幅"].iloc[0]),
                "成交量(手)": self._format_number(stock_data["成交量"].iloc[0] / 100),
                "成交额(万)": self._format_number(stock_data["成交额"].iloc[0] / 10000),
                "更新时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 美化输出
            print("\n" + "="*55)
            print(f"📊 {self.stock_name}({self.stock_code}) 实时行情")
            print("="*55)
            table = PrettyTable()
            table.field_names = ["指标", "数值"]
            for k, v in self.realtime_data.items():
                table.add_row([k, v])
            print(table)
            return self.realtime_data

        except Exception as e:
            print(f"❌ 实时价格获取失败：{str(e)}")
            return None

    def _get_indicator_config(self):
        """根据周期（分钟/日线）匹配对应接口"""
        is_minute = self.period in ["1", "5", "15", "30", "60"]
        
        # 分钟线接口
        if is_minute:
            return {
                "KDJ": ak.stock_zh_a_kdj_min_em,
                "BOLL": ak.stock_zh_a_boll_min_em,
                "RSI": ak.stock_zh_a_rsi_min_em,
                "WR": ak.stock_zh_a_wr_min_em,
                "MACD": ak.stock_zh_a_macd_min_em,
                "MA": ak.stock_zh_a_ma_min_em,
                "CCI": ak.stock_zh_a_cci_min_em,
                "BIAS": ak.stock_zh_a_bias_min_em
            }
        # 日线接口
        else:
            return {
                "KDJ": ak.stock_zh_a_kdj_em,
                "BOLL": ak.stock_zh_a_boll_em,
                "RSI": ak.stock_zh_a_rsi_em,
                "MACD": ak.stock_zh_a_macd_em,
                "MA": ak.stock_zh_a_ma_em
            }

    def get_technical_indicators(self):
        """获取技术指标（自动适配分钟/日线，纯数字代码）"""
        if not self.realtime_data:
            self.get_realtime_price()

        period_text = f"{self.period}分钟" if self.period.isdigit() else "日线"
        print(f"\n📈 {self.stock_name}({self.stock_code}) 技术指标 | {period_text} | {self.adjust}复权")
        print("-"*65)

        indicators = {}
        func_map = self._get_indicator_config()

        for name, func in func_map.items():
            try:
                # 核心修复：技术指标使用【纯数字代码】
                df = func(symbol=self.code_plain, period=self.period, adjust=self.adjust)
                if df.empty:
                    print(f"⏹️  {name}：无数据")
                    continue

                # 取最新5条，格式化数据
                latest = df.tail(5).copy()
                # 数值保留2位小数
                for col in latest.columns:
                    if col not in ["date", "time"]:
                        latest[col] = latest[col].apply(self._format_number)
                indicators[name] = latest

                # 输出表格
                print(f"\n🔹 {name} 指标（最新5条）")
                table = PrettyTable()
                table.field_names = latest.columns.tolist()
                for _, row in latest.iterrows():
                    table.add_row(row.tolist())
                print(table)

            except Exception as e:
                print(f"⚠️ {name} 获取失败：{str(e)}")

        return indicators

    def run(self):
        """执行完整流程"""
        self.get_realtime_price()
        self.get_technical_indicators()
        print("\n" + "="*55)
        print("✅ 所有数据获取完成！")
        print("="*55)


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # ========== 自定义参数 ==========
    TARGET_STOCK = "sh600000"  # 浦发银行 | 平安银行: sz000001 | 贵州茅台: sh600519
    TIME_PERIOD = "1"           # 周期：1/5/15/30/60(分钟) / daily(日线)
    PRICE_ADJUST = "qfq"        # 前复权（推荐）

    # 启动
    fetcher = StockDataFetcher(TARGET_STOCK, TIME_PERIOD, PRICE_ADJUST)
    fetcher.run()