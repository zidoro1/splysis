import os
#from readline import get_completer_delims
import sys
import cv2
import numpy as np

from get_match import getCrd, getExist, getNum
from get_scene import getResult, getGraph, fixCsv
from get_color import cvtGry, getCrdColor

from set_class import MyCudaStruct

def  main(in_path :str, out_path="C:/Users/ntkke/ProjectSplaly/output/csv"):

    dic_templ_dir = {
        "StoF"   : "C:/Users/ntkke/ProjectSplaly/data/images/mark/mark_StoF_gry",
        "result" : "C:/Users/ntkke/ProjectSplaly/data/images/mark/mark_battlelog_gry",
        "graph"  : "C:/Users/ntkke/ProjectSplaly/data/images/mark/mark_graph_gry",
        "player" : "C:/Users/ntkke/ProjectSplaly/data/images/player/player_game_gry",
        "sp"     : "C:/Users/ntkke/ProjectSplaly/data/images/num/num_sp_gry_m",
        "buki"   : "C:/Users/ntkke/ProjectSplaly/data/images/buki",
        "life"   : "C:/Users/ntkke/ProjectSplaly/data/images/mark/mark_life_gry"
    }

    # テンプレート画像のpathを一括管理
    dic_fname_path = {"StoF": "v1", "result": "v2", "graph": "v3", "player": "v4", "sp": "v5", "buki": "v6", "mark": "v7"}
    for key, value in dic_templ_dir.items(): 
        dic_fname_path[key] = {os.path.splitext(f)[0]: os.path.join(value, f) for f in os.listdir(value) if os.path.isfile(os.path.join(value, f))}

    # デバイスメモリを確保
    frame_gpu_src      = cv2.cuda_GpuMat()
    frame_gpu_gry      = cv2.cuda_GpuMat()
    frame_gpu_trim     = cv2.cuda_GpuMat()
    gpu_dst            = cv2.cuda_GpuMat()

    mark_gpu_result    = cv2.cuda_GpuMat(cv2.imread(dic_fname_path["result"]["start_log"],  cv2.IMREAD_GRAYSCALE))
    mark_gpu_graph     = cv2.cuda_GpuMat(cv2.imread(dic_fname_path["graph"]["start_graph"], cv2.IMREAD_GRAYSCALE))
    mark_gpu_death     = cv2.cuda_GpuMat(cv2.imread(dic_fname_path["life"]["death"],        cv2.IMREAD_GRAYSCALE))
    mark_gpu_start     = cv2.cuda_GpuMat(cv2.imread(dic_fname_path["StoF"]["start_59"],     cv2.IMREAD_GRAYSCALE))
    mark_gpu_finish    = cv2.cuda_GpuMat(cv2.imread(dic_fname_path["StoF"]["finish_pin"],   cv2.IMREAD_GRAYSCALE))

    templ_gpu_result   = MyCudaStruct()
    templ_gpu_graph    = MyCudaStruct()
    templ_gpu_player   = MyCudaStruct()
    templ_gpu_sp       = MyCudaStruct()
    templ_gpu_buki     = MyCudaStruct()

    # gpuメモリに全テンプレート画像を一括転送
    # ファイル名（拡張子なし）をKeyとして、読み込んだ値（テンプレート画像）をvalueとする辞書
    templ_img = {}
    for fname, path in dic_fname_path["result"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_result.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["graph"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_graph.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["player"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_player.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["sp"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_sp.set(fname, templ_img[fname])
    for fname, path in dic_fname_path["buki"].items():
        templ_img[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templ_gpu_buki.set(fname, templ_img[fname])

    # フラグ
    count            = 0
    flag_result      = 0
    flag_graph       = 0
    flag_time_start  = 0

    # 誤差・補正
    corr   = {
        "a": (13, 55), "b": (13, 55), "y": (13, 55), "x": (13, 55), # icon座標とplayer座標の間の誤差
        "u": (12, 60), "r": (4,  54), "d": (12, 60), "l": (4,  54),
        "sp": 95,      # trimmingの補正値
    }

    trim = {
        "b": [  80,  150,  160,  370],  # バトル戦績画面の目印
        "g": [ 980, 1080,    0,  100],  # グラフ画像フラグの範囲。左端の100x100
        "s": [  20,  111,  880, 1030],  # 計測開始フラグである_:59の範囲
        "f": [ 500,  700, 1430, 1750]   # 計測終了フラグであるFinishの範囲
    }

    trim_lamp = {
        "a": [45, 100,  520,  612], # 50-90 は採取するテンプレ画像の範囲
        "b": [45, 100,  612,  701], # bのSPとブキアイコン
        "y": [45, 100,  701,  791], # yのSPとブキアイコン
        "x": [45, 100,  791,  878],
        "u": [45, 100, 1042, 1129],
        "r": [45, 100, 1129, 1219],
        "d": [45, 100, 1219, 1308],
        "l": [45, 100, 1308, 1400]
    } 

    trim_sp = {
        "a": [  10,   60,  544,  544+corr["sp"]], 
        "b": [  10,   60,  636,  636+corr["sp"]],
        "y": [  10,   60,  725,  725+corr["sp"]],
        "x": [  10,   60,  816,  816+corr["sp"]],
        "u": [  10,   60, 1066, 1066+corr["sp"]],
        "r": [  10,   60, 1153, 1153+corr["sp"]],
        "d": [  10,   60, 1243, 1243+corr["sp"]],
        "l": [  10,   60, 1332, 1332+corr["sp"]]
    }

    # プレイヤーの時系列データ
    p_is   = {"a":True, "b":True, "y":True, "x":True, "u": True, "r": True, "d": True, "l":True}
    p_life = {"a":   1, "b":   1, "y":   1, "x":   1, "u":    1, "r":    1, "d":    1, "l":   1}
    p_x    = {"a":   0, "b":   0, "y":   0, "x":   0, "u":    0, "r":    0, "d":    0, "l":   0}
    p_y    = {"a":   0, "b":   0, "y":   0, "x":   0, "u":    0, "r":    0, "d":    0, "l":   0}
    p_sp   = {"a":   0, "b":   0, "y":   0, "x":   0, "u":    0, "r":    0, "d":    0, "l":   0}
    p_val  = {"a":   0, "b":   0, "y":   0, "x":   0, "u":    0, "r":    0, "d":    0, "l":   0}

    fileobj = None
    cap = cv2.VideoCapture(in_path)
    # 動画処理開始。のちのちキャプチャーボードの直接解析に移行する
    if not cap.isOpened():
        raise Exception("cap.isOpened() is False")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        count += 1
        resized_frame = cv2.resize(frame,(1920//2, 1080//2))
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", resized_frame)
        cv2.waitKey(1)

        # カラー画像、グレースケール画像を用意。それぞれをGPUメモリに転送する
        frame_gpu_src.upload(frame)
        frame_gry = cvtGry(frame, pattern="lightness")
        frame_gpu_gry.upload(frame_gry)

        # 現在のframeがリザルト画面である場合、一回だけリザルト画面の情報を採取
        if  flag_result == 0:
            frame_gpu_trim.upload(frame_gry[trim["b"][0]:trim["b"][1], trim["b"][2]:trim["b"][3]])
            is_result, _, _, _ = getCrd(frame_gpu_trim, mark_gpu_result, gpu_dst, pattern="mark")
            if not is_result:
                print("\r{:>8}: Not   [Result]".format(count), end="")
                continue
            else:
                print("\r{:>8}: Start [Result]".format(count))
                ls_result = getResult(frame_gry)
                print("\r{:>8}: End   [Result]".format(count))
                # リザルト画面から"ymdhm".csvがすでにある場合、フラグリセット
                csv_file = ls_result[0]+".csv"
                csv_path = os.path.join(out_path, csv_file)
                if not os.path.isfile(csv_path):
                    flag_result = 1
                    print("CSV Name: \"{}\"".format(csv_file))
                else:
                    print("\"{}\" already exists\n".format(csv_file))
                    continue
        else:
            pass

        # 現在のframeがグラフ画面である場合、一回だけグラフ画面の情報を採取
        if  flag_graph == 0:
            frame_gpu_trim.upload(frame_gry[trim["g"][0]:trim["g"][1], trim["g"][2]:trim["g"][3]])
            is_graph, _, _, _ = getCrd(frame_gpu_trim, mark_gpu_graph, gpu_dst, pattern="mark")
            if not is_graph:
                print("\r{:>8}: Not   [Graph]".format(count), end="")
                continue
            else:
                print("\r{:>8}: Start [Graph]".format(count))
                # バトルメモリーの再生が始まり次第、ファイルオープン
                fileobj = open(csv_path, mode="x", encoding="utf_8")
                fileobj.write("{}\n".format("\n".join(ls_result)))
                fileobj.write(
                    "a_x,a_y,a_sp,a_life,b_x,b_y,b_sp,b_life,y_x,y_y,y_sp,y_life,x_x,x_y,x_sp,x_life,u_x,u_y,u_sp,u_life,r_x,r_y,r_sp,r_life,d_x,d_y,d_sp,d_life,l_x,l_y,l_sp,l_life\n"
                )
                # *** ここにグラフ画面の処理が入る ***
                flag_graph = 1
                print("\r{:>8}: End   [Graph]".format(count))
        else:
            pass

        # 対戦の時間掲示が-:00から-:59になった瞬間1回だけ以下を実行
        if flag_time_start == 0:
            frame_gpu_trim.upload(frame_gry[trim["s"][0]:trim["s"][1], trim["s"][2]:trim["s"][3]])
            is_start, _, _, _ = getCrd(frame_gpu_trim, mark_gpu_start, gpu_dst, pattern="mark")
            if not is_start:
                print("\r{:>8}: Not   [Game]".format(count), end="")
                continue
            else:
                print("\r{:>8}: Start [Game]".format(count))
                flag_time_start = 1
                
        else:
            print("\r{:>8}: In    [Game]".format(count), end="")

        # 試合中データの採取 
        for key in p_is.keys():
            # プレイヤーの生死判定( 生:0 or 死:1 )
            # ******* プレイヤーをリンクしていないので、めんどくさい *******
            frame_gpu_trim.upload(frame_gry[trim_lamp[key][0]:trim_lamp[key][1], trim_lamp[key][2]:trim_lamp[key][3]])
            p_life[key], _ , _, _ = getCrd(frame_gpu_trim, mark_gpu_death, gpu_dst, pattern="mark")
            # プレイヤーのSP判定
            frame_gpu_trim.upload(frame_gry[trim_sp[key][0]:trim_sp[key][1], trim_sp[key][2]:trim_sp[key][3]])
            p_sp[key], p_val[key] = getNum(frame_gpu_trim, templ_gpu_sp, gpu_dst, pattern="sp")
            # プレイヤー位置座標判定
            p_is[key], p_x[key], p_y[key], _ = getCrd(frame_gpu_gry, templ_gpu_player.get(key), gpu_dst, pattern="mark")

        # nanが連続したとき、finish画像とマッチングしたら、計測終了
        if not any(p_is.values()) and flag_time_start == 1:
            frame_gpu_trim.upload(frame_gry[trim["f"][0]:trim["f"][1], trim["f"][2]:trim["f"][3]])
            is_fin, _, _, _ = getCrd(frame_gpu_trim, mark_gpu_finish, gpu_dst, pattern="mark")
            if is_fin:
                # フラグのリセット
                flag_result      = 0
                flag_graph       = 0
                flag_time_start  = 0
                fileobj.close()
                fixCsv(csv_path)
                print("\r{:>8}: End   [Game]    \n".format(count))
                continue
            else:
                print("\r{:>8}: Not   [Game End]".format(count), end="")

        # データの書き込み
        for key in p_x.keys():
            fileobj.write("{},{},{},{}".format(p_x[key]-corr[key][0], p_y[key]-corr[key][1], p_sp[key], p_life[key]))
            fileobj.write("\n") if key=="l" else fileobj.write(",")

    print("\r{:>8}: End   [All Frame]\n".format(count))
    if fileobj is not None and not fileobj.closed:
        fileobj.close()
        fixCsv(csv_path)
    cap.release()
    cv2.destroyAllWindows()

###################################################################################
# コマンドライン実行

if __name__ == "__main__":
    # 入力動画と出力先
    in_dir1    = "C:/Users/ntkke/Videos/Spla3Video/BattleMemory"
    in_dir2    = "C:/Users/ntkke/ProjectSplaly/data/sample_video"
    in_path   = os.path.join(in_dir2, sys.argv[1])

    import timeit
    time_func = timeit.timeit("main(in_path)", globals=globals(), number=1)
    print("Time: {:>8}".format(time_func))