import akshare as ak
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from rl_trading_env import TradingEnv
from indicator_strategy import IndicatorStrategy
import os

def prepare_data(symbol="002156"):
    strategy = IndicatorStrategy(symbol=symbol)
    df = strategy.fetch_data()
    if df is None: return None
    df = strategy.calculate_indicators(df)
    return df

def train_rl_model(symbol="002156", timesteps=200000):
    df = prepare_data(symbol)
    if df is None: return
    
    # 使用更近期的数据训练 (2023-2025)
    train_df = df[(df['datetime'] >= '2023-01-01') & (df['datetime'] <= '2025-12-31')]
    print(f"训练数据范围: {train_df['datetime'].min()} 至 {train_df['datetime'].max()}")
    print(f"训练数据条数: {len(train_df)}")
    
    # 创建环境
    env = TradingEnv(train_df)
    
    # 初始化模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )
    
    print(f"\n--- 正在为 {symbol} 训练强化学习策略 (V23优化版) ---")
    print(f"训练步数: {timesteps}")
    model.learn(total_timesteps=timesteps)
    
    # 保存模型
    model_name = f"ppo_trading_{symbol}"
    model.save(model_name)
    print(f"\n模型已保存为 {model_name}.zip")
    return model

if __name__ == "__main__":
    train_rl_model("002156", timesteps=200000)
