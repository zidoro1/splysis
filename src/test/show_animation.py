import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# 指定したルール・ステージの画像のpathを与える関数
def stage_check(rule_name: str, stage_name: str):
    rule_dic ={
        "asari"  : "../../data/images/stage/stage_map_asari",
        "area"   : "../../data/images/stage/stage_map_area",
        "hoko"   : "../../data/images/stage/stage_map_hoko",
        "yagura" : "../../data/images/stage/stage_map_yagura" 
    }
    stage_dic = {
        "amabi"    : "amabi.png",
        "baigai"   : "baigai.jpg",
        "chozame"  : "chozame.png",
        "gonzui"   : "gonzui.png",
        "hirame"   : "hirame.png",
        "kinme"    : "kinme.png",
        "konbu"    : "konbu.png",
        "kusaya"   : "kusaya.png",
        "mahimahi" : "mahimahi.png",
        "manta"    : "manta.png",
        "masaba"   : "masaba.png",
        "mategai"  : "mategai.png",
        "namero"   : "namero.png",
        "nanpura"  : "nanpura.png",
        "negitoro" : "negitoro.png",
        "ohyo"     : "ohyo.png",
        "sumesi"   : "sumesi.png",
        "takaasi"  : "takaasi.png",
        "tarapo"   : "tarapo.png",
        "yagara"   : "yagara.png",
        "yunohana" : "yunohana.png",
        "zatou"    : "zatou.png"
    }

    if rule_name in rule_dic:
        rule_dir = rule_dic[rule_name]
    else:
        raise Exception("error:in rule_check(). rule_name is not in rule_dic")
    
    if stage_name in stage_dic:
        stage_file = stage_dic[stage_name]
    else:
        raise Exception("error:in stage_check(). stage_name is not in stage_dic")
    
    stage_path = os.path.join(rule_dir, stage_file)
    return stage_path
    

# CSVのアニメーション表示
def show_animation(in_path: str, out_path: str):

    # 背景画像の特定と取得
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        stage_name = lines[1].rstrip('\n').strip()
        rule_name  = lines[3].rstrip('\n').strip()
    #print("stage_name and rule_name:", stage_name, rule_name)
    stage_path = stage_check(rule_name, stage_name)
    stage_img  = Image.open(stage_path)

    dic_data = {
        'a_x', 'a_y', 'a_sp', 'a_life',
        'b_x', 'b_y', 'b_sp', 'b_life',
        'y_x', 'y_y', 'y_sp', 'y_life',
        'x_x', 'x_y', 'x_sp', 'x_life',
        'u_x', 'u_y', 'u_sp', 'u_life',
        'r_x', 'r_y', 'r_sp', 'r_life',
        'd_x', 'd_y', 'd_sp', 'd_life',
        'l_x', 'l_y', 'l_sp', 'l_life'
    }
    # データの取得
    df_data = pd.read_csv(in_path, header=12)
    len_time = len(df_data)    # 時間幅
    ls_time  = range(len_time)

    # グラフ領域の作成
    fig = plt.figure(figsize=(8.0, 5.0), constrained_layout=True)
    gs = gridspec.GridSpec(20, 12, figure=fig)

    # ax1は座標
    ax1 = fig.add_subplot(gs[0:15, 0:8])
    # ax2はSP
    ax2 = fig.add_subplot(gs[16:20, 0:8])
    ticks_top = np.linspace(0, 100, 11)

    def update(frame):
        ax1.cla()
        ax1.set_title(rule_name + " " + stage_name, color="k")
        ax1.set_xlim([0, 1920])
        ax1.set_ylim([1080, 0])
        ax1.imshow(stage_img, extent=[0, 1920, 1080, 0], alpha=0.7)

        ax1.plot(df_data.a_x[frame], df_data.a_y[frame], marker="o", markersize=4, color="none", markeredgecolor="r")
        ax1.plot(df_data.b_x[frame], df_data.b_y[frame], marker="^", markersize=4, color="none", markeredgecolor="r")
        ax1.plot(df_data.y_x[frame], df_data.y_y[frame], marker="s", markersize=3.7, color="none", markeredgecolor="r")
        ax1.plot(df_data.x_x[frame], df_data.x_y[frame], marker="D", markersize=3.5, color="none", markeredgecolor="r")
        ax1.plot(df_data.u_x[frame], df_data.u_y[frame], marker="o", markersize=4, color="none", markeredgecolor="b")
        ax1.plot(df_data.r_x[frame], df_data.r_y[frame], marker="^", markersize=4, color="none", markeredgecolor="b")
        ax1.plot(df_data.d_x[frame], df_data.d_y[frame], marker="s", markersize=3.7, color="none", markeredgecolor="b")
        ax1.plot(df_data.l_x[frame], df_data.l_y[frame], marker="D", markersize=3.5, color="none", markeredgecolor="b")


        ax2.cla()
        ax2.set_xlim([0, len_time])
        ax2.set_ylim([-10, 100])
        ax2.grid(which="minor", alpha=0.3)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("SP")
        ax2.set_yticks(ticks_top, minor=True)

        ax2.plot(ls_time, df_data.a_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.b_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.y_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.x_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.u_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.r_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.d_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(ls_time, df_data.l_sp, linewidth=0.5, linestyle="dashed", alpha=0.5, color="k")
        ax2.plot(frame, df_data.a_sp[frame], marker="o", markersize=3, color="none", markeredgecolor="r")
        ax2.plot(frame, df_data.b_sp[frame], marker="^", markersize=3, color="none", markeredgecolor="r")
        ax2.plot(frame, df_data.y_sp[frame], marker=",", markersize=3, color="none", markeredgecolor="r")
        ax2.plot(frame, df_data.x_sp[frame], marker="x", markersize=3, color="none", markeredgecolor="r")
        ax2.plot(frame, df_data.u_sp[frame], marker="o", markersize=3, color="none", markeredgecolor="b")
        ax2.plot(frame, df_data.r_sp[frame], marker="^", markersize=3, color="none", markeredgecolor="b")
        ax2.plot(frame, df_data.d_sp[frame], marker=",", markersize=3, color="none", markeredgecolor="b")
        ax2.plot(frame, df_data.l_sp[frame], marker="x", markersize=3, color="none", markeredgecolor="b")
    
    anim = FuncAnimation(fig, update, frames=len_time, interval=25) # intervalの単位はミリ秒
    plt.rcParams["animation.ffmpeg_path"] = "C:/Program Files/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
    anim.save(out_path, writer="ffmpeg", fps=4)
    plt.show()
    plt.close()

###################################################################################
# コマンドライン実行
if __name__ == "__main__":
    in_dir   = "../../output/csv"
    in_path  = os.path.join(in_dir, sys.argv[1] + ".csv")  
    out_dir  = "../../output/anime"
    out_path = os.path.join(out_dir, sys.argv[1] + ".mp4")
    import timeit
    time_func = timeit.timeit("show_animation(in_path, out_path)", globals=globals(), number=1)
    print(time_func)