import sys
import os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

def showHist(in_path:str, out_path:str, width=1, pattern="1"):
    pattern = {}
    width   = {}

    # データの取得
    df_data = pd.read_csv(in_path, header=12)
    #del df_data["Name"], df_data["Ticket"], df_data["Cabin"] # 今回使わない列データは削除する
    plt.hist(df_data["d_life"].dropna(), bins = 100, range = (0.5,1),color = 'Blue')
    plt.show()




###################################################################################
# コマンドライン実行
if __name__ == "__main__":
    in_dir   = "C:/Users/ntkke/ProjectSplaly/output/csv"
    in_path  = os.path.join(in_dir, sys.argv[1] + ".csv")  
    out_dir  = "C:/Users/ntkke/ProjectSplaly/output/aanalysis"
    out_path = os.path.join(out_dir, sys.argv[1])
    showHist(in_path, out_path)