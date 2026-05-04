import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

def calculate_indicators(df):
    # 确保按时间排序
    df = df.sort_values('日期')
    
    # MA
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()

    # MACD
    exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['dif'] = exp1 - exp2
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    # KDJ
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # WR
    df['wr'] = (high_9 - df['收盘']) / (high_9 - low_9) * -100

    # BIAS
    df['bias6'] = (df['收盘'] - df['收盘'].rolling(6).mean()) / df['收盘'].rolling(6).mean() * 100

    # BOLL
    df['boll_mid'] = df['收盘'].rolling(20).mean()
    df['boll_std'] = df['收盘'].rolling(20).std()
    df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_low'] = df['boll_mid'] - 2 * df['boll_std']

    # CCI
    tp = (df['最高'] + df['最低'] + df['收盘']) / 3
    ma_tp = tp.rolling(20).mean()
    md_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - ma_tp) / (0.015 * md_tp)

    # 量比
    df['vol_ma5'] = df['成交量'].rolling(5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']

    # 神奇九转
    df['close_ref4'] = df['收盘'].shift(4)
    df['up_count'] = (df['收盘'] > df['close_ref4']).astype(int)
    df['dn_count'] = (df['收盘'] < df['close_ref4']).astype(int)
    
    def get_consecutive_counts(series):
        counts = []
        cur = 0
        for val in series:
            if val == 1: cur += 1
            else: cur = 0
            counts.append(cur)
        return counts
    
    df['magic_up'] = get_consecutive_counts(df['up_count'])
    df['magic_dn'] = get_consecutive_counts(df['dn_count'])

    return df

def train_model():
    print("正在加载数据...")
    df = pd.read_csv("data.csv")
    df['日期'] = pd.to_datetime(df['日期'])
    
    print("正在计算技术指标...")
    # 对每只股票分别计算指标
    df = df.groupby('symbol', group_keys=False).apply(calculate_indicators)
    df = df.fillna(0)
    
    # 准备特征
    feature_cols = [
        'dif', 'dea', 'macd', 'k', 'd', 'j', 'rsi', 'wr', 'bias6', 
        'boll_up', 'boll_low', 'cci', 'vol_ratio', 'magic_up', 'magic_dn',
        'MA5', 'MA10', 'MA20'
    ]
    
    # 标签：未来3天涨幅 > 2%
    df['label'] = (df.groupby('symbol')['收盘'].shift(-3) / df['收盘'] > 1.02).astype(int)
    
    # 筛选训练集：2020年1月1日 至 2024年12月31日
    train_df = df[(df['日期'] >= '2020-01-01') & (df['日期'] <= '2024-12-31')].dropna()
    
    if train_df.empty:
        print("错误：训练集为空，请检查 data.csv 是否包含指定日期范围内的数据")
        return

    X = train_df[feature_cols]
    y = train_df['label']
    
    print(f"训练样本数: {len(X)}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 构建堆叠模型 (Stacking)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    ]
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=3
    )
    
    print("开始训练堆叠模型 (这可能需要一些时间)...")
    clf.fit(X_scaled, y)
    
    # 保存权重
    model_data = {
        'model': clf,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, 'model_weights.joblib')
    print("模型权重已保存至 model_weights.joblib")

if __name__ == "__main__":
    train_model()
