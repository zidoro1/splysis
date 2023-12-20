import os
import sys
import cv2
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

from set_class import MyCudaStruct
from get_match import getCrd, getExist, getNum

# リザルト画面の情報を採取・保存する関数
def getResult(src_gry):
    # テンプレート画像のpathを二次元の辞書型にして一括管理
    dic_templ_dir = {
        "stage" : "../../data/templates/stage/stage_battlelog_gry",
        "match" : "../../data/templates/match/match_battlelog_gry",
        "rule"  : "../../data/templates/rule/rule_battlelog_gry",
        "mark"  : "../../data/templates/mark/mark_battlelog_gry",
        "buki"  : "../../data/templates/buki/buki_battlelog_gry",
        "sp"    : "../../data/templates/sp/sp_battlelog_gry",
        "ymdhm" : "../../data/templates/num/num_battlelog_gry/ymdhm",
        "np"    : "../../data/templates/num/num_battlelog_gry/np",
        "score" : "../../data/templates/num/num_battlelog_gry/score"
    }
    dic_fname_path = {"stage": "v1", "match": "v2", "rule": "v3", "mark": "v4", "buki": "v5", "sp": "v6", "ymdhm": "v7", "np": "v8", "score": "v9"}
    for key, value in dic_templ_dir.items():
        # num_0.png -> "num_0": "C:~/~~/~/num_0.png" 
        dic_fname_path[key] = {os.path.splitext(f)[0]: os.path.join(value, f) for f in os.listdir(value) if os.path.isfile(os.path.join(value, f))}
    
    # ブキのpathだけは別の方法で取得する
    buki_kind_dir = {k: os.path.join(dic_templ_dir["buki"], k) for k in os.listdir(dic_templ_dir["buki"])}
    for kind, path_k in buki_kind_dir.items():
        for fname in os.listdir(path_k):
            buki = "_".join([kind, os.path.splitext(fname)[0]])
            path  = os.path.join(path_k, fname)
            dic_fname_path["buki"][buki] = path

    # GPUメモリを確保。
    # ソース画像(1)・切り取り画像(1)・0～9の数字(10)＋記号(1,2)・ステージ()・マッチ(4)・ルール(5)・目印(3)・ブキ()・sp(19)
    frame_gpu_trim  = cv2.cuda_GpuMat()
    frame_gpu_dst   = cv2.cuda_GpuMat()
    templ_gpu_stage = MyCudaStruct()
    templ_gpu_match = MyCudaStruct()
    templ_gpu_rule  = MyCudaStruct()
    templ_gpu_mark  = MyCudaStruct()
    templ_gpu_buki  = MyCudaStruct()
    templ_gpu_sp    = MyCudaStruct()
    templ_gpu_ymdhm = MyCudaStruct()
    templ_gpu_np    = MyCudaStruct()
    templ_gpu_score = MyCudaStruct()

    # ファイル名（拡張子なし）をKeyとして、読み込んだ値（テンプレート画像）をvalueとする辞書
    templ_img = {}
    for fname, path in dic_fname_path["stage"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_stage.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["match"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_match.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["rule"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_rule.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["mark"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_mark.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["buki"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_buki.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["sp"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_sp.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["ymdhm"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_ymdhm.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["np"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_np.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["score"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_score.set(fname, templ_img[fname])

    # テンプレートマッチングを行う画像範囲
    trim = {
        "h" : [  59,  180,  800, 1500],  # ヘッダー
        "y" : [  70,  110,  830, 1090],  # ヘッダーの日時領域
        "r" : [ 330, 1080,  900, 1800],  # scoreが表示される領域
        "0" : [ 420,  485,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700], #   0,   1,        2,        3,      4,      5,       6,       7,       8
        "1" : [ 485,  550,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700], # y_s, y_e, x_icon_s, x_icon_e, x_np_s, x_np_e, x_sc1_e, x_sc2_e, x_sc3_e
        "2" : [ 550,  620,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700], 
        "3" : [ 615,  690,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700], 
        "4" : [ 785,  853,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700],
        "5" : [ 853,  920,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700],
        "6" : [ 920,  983,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700],
        "7" : [ 983, 1050,  950, 1130, 1300, 1440, 1505, 1570, 1635, 1700],
    }

    frame_gpu_trim.upload(src_gry[trim["y"][0]:trim["y"][1], trim["y"][2]:trim["y"][3]])
    # 年月日・時分判定
    ymdhm, _ = getNum(frame_gpu_trim, templ_gpu_ymdhm, frame_gpu_dst, pattern="ymdhm")

    # ステージ名判定・マッチ区分判定・ルール判定・チームの勝敗判定
    frame_gpu_trim.upload(src_gry[trim["h"][0]:trim["h"][1], trim["h"][2]:trim["h"][3]])
    stage_is, stage_ls = getExist(frame_gpu_trim, templ_gpu_stage, frame_gpu_dst, pattern="stage")
    match_is, match_ls = getExist(frame_gpu_trim, templ_gpu_match, frame_gpu_dst, pattern="match")
    rule_is,   rule_ls = getExist(frame_gpu_trim, templ_gpu_rule,  frame_gpu_dst, pattern="rule")
    stage = stage_ls[0] if stage_is else input("stage: ")
    match = match_ls[0] if match_is else input("match: ")
    rule  = rule_ls[0]  if rule_is  else input("rule: ")

    # 各プレイヤーのスコア情報の採取
    buki        = ["nan"]*8
    who         = ["others"]*8
    score_np    = [np.nan]*8
    score_kill  = [np.nan]*8
    score_death = [np.nan]*8
    score_sp    = [np.nan]*8
    for i in range(8): # ８はプレイヤー数。つまり、調べる画像の帯数
        ci = str(i)
        # 自機の確認
        frame_gpu_trim.upload(src_gry[trim[ci][0]:trim[ci][1], trim[ci][2]:trim[ci][3]])
        self_is, _, _, _ = getCrd(frame_gpu_trim, templ_gpu_mark.get("myself"), frame_gpu_dst, pattern="mark")
        if self_is:
            who[i] = "myself"
        # ブキ判定
        buki_is, buki_ls   = getExist(frame_gpu_trim, templ_gpu_buki,  frame_gpu_dst, pattern="buki")
        if buki_is:
            buki[i] = buki_ls[0]
        # 各スコア数値の判定
        frame_gpu_trim.upload(src_gry[trim[ci][0]:trim[ci][1], trim[ci][4]:trim[ci][5]])
        score_np[i],    _ = getNum(frame_gpu_trim, templ_gpu_np, frame_gpu_dst, pattern="np")
        frame_gpu_trim.upload(src_gry[trim[ci][0]:trim[ci][1], trim[ci][6]:trim[ci][7]])
        score_kill[i],  _ = getNum(frame_gpu_trim, templ_gpu_score, frame_gpu_dst, pattern="score")
        frame_gpu_trim.upload(src_gry[trim[ci][0]:trim[ci][1], trim[ci][7]:trim[ci][8]])
        score_death[i], _ = getNum(frame_gpu_trim, templ_gpu_score, frame_gpu_dst, pattern="score")
        frame_gpu_trim.upload(src_gry[trim[ci][0]:trim[ci][1], trim[ci][8]:trim[ci][9]])
        score_sp[i],    _ = getNum(frame_gpu_trim, templ_gpu_score, frame_gpu_dst, pattern="score")

    # プレイヤーのブキ種・プレイヤー名・スコア判定
    p1 = "{},{},{},{},{},{},win" .format(buki[0], who[0], score_np[0], score_kill[0], score_death[0], score_sp[0])
    p2 = "{},{},{},{},{},{},win" .format(buki[1], who[1], score_np[1], score_kill[1], score_death[1], score_sp[1])
    p3 = "{},{},{},{},{},{},win" .format(buki[2], who[2], score_np[2], score_kill[2], score_death[2], score_sp[2])
    p4 = "{},{},{},{},{},{},win" .format(buki[3], who[3], score_np[3], score_kill[3], score_death[3], score_sp[3])
    p5 = "{},{},{},{},{},{},lose".format(buki[4], who[4], score_np[4], score_kill[4], score_death[4], score_sp[4])
    p6 = "{},{},{},{},{},{},lose".format(buki[5], who[5], score_np[5], score_kill[5], score_death[5], score_sp[5])
    p7 = "{},{},{},{},{},{},lose".format(buki[6], who[6], score_np[6], score_kill[6], score_death[6], score_sp[6])
    p8 = "{},{},{},{},{},{},lose".format(buki[7], who[7], score_np[7], score_kill[7], score_death[7], score_sp[7])

    result_all = list(map(str, [ymdhm, stage, match, rule, p1, p2, p3, p4, p5, p6, p7, p8]))

    return result_all

def fixCsv(csv_path):
 a=0
def getGraph():
    a=0




###################################################################################
# コマンドライン実行

if __name__ == "__main__":
    in_dir_color = "../../data/sample_frame/sample_result"
    in_dir_gry   = "../../data/sample_frame/sample_result_gry"
    """
    files_gry = [
        os.path.join(in_dir_gry, f) for f in os.listdir(in_dir_gry) if os.path.isfile(os.path.join(in_dir_gry, f))
    ]

    for file_gry in files_gry:
        frame_gry      = cv2.imread(file_gry, cv2.IMREAD_GRAYSCALE)
        print(getResult(frame_gry).replace("\n", " "))
    """
    frame_gry      = cv2.imread(os.path.join(in_dir_gry, "01.png"), cv2.IMREAD_GRAYSCALE)
    print(getResult(frame_gry))

