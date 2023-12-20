import os
import sys
import cv2
import numpy as np
import pandas as pd

csv_path = "../../output/csv/202312161944.csv"
# CSVファイルの読み込み
df = pd.read_csv(csv_path, header=12)
df_data = df

ls_x    = [   "a_x",    "b_x",    "y_x",    "x_x",    "u_x",    "r_x",    "d_x",    "l_x"]
ls_life = ["a_life", "b_life", "y_life", "x_life", "u_life", "r_life", "d_life", "l_life"]

for col1, col2 in zip(ls_x, ls_life):
    df_data[col1] = df_data[col1].notna().astype(int)
    df_data[col2] = df_data[col2]*1
    
print(df_data[ls_x + ls_life].corr())


#print(df_data)
#print(df_data.isnull().sum())

# 欠損値と相関のある列AとBのペアを見つける
correlated_pairs = []
for col1 in df_data.columns:
    for col2 in df_data.columns:
        if col1 != col2:
            correlation = df[col1].corr(df[col2])
            if pd.isna(correlation):
                continue
            if correlation > 0.8:  # 相関が高いと仮定
                correlated_pairs.append((col1, col2))
print(correlated_pairs)
"""
if pd.isna(correlation):
    continue
if correlation > 0.8:  # 相関が高いと仮定
    correlated_pairs.append((col1, col2))

# 列の整理と入れ替え
for col1, col2 in correlated_pairs:
    # 列Bが1である行だけを抽出
    subset = df[df[col2] == 1]

    # 列Aの欠損値を列Bが1である行の平均値で埋める
    df[col1].fillna(subset[col1].mean(), inplace=True)

    # 列Bの1の行を削除
    df = df[df[col2] != 1]

# CSVファイルに書き込み
df.to_csv('edited_data.csv', index=False)
"""