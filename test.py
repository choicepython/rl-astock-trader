#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Yang
# @Project : 量化
import akshare as ak
import pandas as pd

# 获取日K线
# df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20230101")
df = ak.stock_zh_a_hist_sina(symbol="600519", period="daily", start_date="20240101")

print(df.head())