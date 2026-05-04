import akshare as ak
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from rl_trading_env import TradingEnv
from indicator_strategy import IndicatorStrategy
import os

def prepare_data(symbol="002156"):
    # 复用 IndicatorStrategy 的指标计算逻辑
    strategy = IndicatorStrategy(symbol=symbol)
    df = strategy.fetch_data()
    if df is None: return None
    df = strategy.calculate_indicators(df)
    return df

def train_rl_model(symbol="002156"):
    df = prepare_data(symbol)
    if df is None: return
    
    # 划分训练集 (2020-2024)
    train_df = df[(df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2024-12-31')]
    
    # 创建环境
    env = TradingEnv(train_df)
    
    # 初始化模型
    model = PPO("MlpPolicy", env, verbose=1)
    
    print(f"--- 正在为 {symbol} 训练强化学习策略 ---")
    model.learn(total_timesteps=100000)
    
    # 保存模型
    model_name = f"ppo_trading_{symbol}"
    model.save(model_name)
    print(f"模型已保存为 {model_name}")
    return model

if __name__ == "__main__":
    # 先为一只标杆股票训练
    train_rl_model("002156")
