import os
import sys
import cv2
import numpy as np

# プレイヤー座標を採取する関数
def getCrdCpu(img, templ, pattern):
    
    threshold = {"player": 0.85, "mark": 0.75, "buki": 0.8}

    result  =  cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_idx = cv2.minMaxLoc(result)

    if max_val >= threshold[pattern]:
        ret = True
        x, y = max_idx
    else:
        ret = False
        x = np.nan
        y = np.nan

    return ret, x, y, max_val

# 複数のテンプレート画像の辞書とマッチングして、マッチングしたkeyだけを返す関数
def getExistCpu(img, dic_templ: dict, pattern):

    # テンプレートマッチングの許容閾値
    threshold = {"stage": 0.7, "match": 0.7, "rule": 0.7, "mark": 0.7, "buki": 0.7}
    
    ls_exist  = []
    max_score = 0
    for key in dic_templ.keys():
        result  =  cv2.matchTemplate(img, dic_templ[key], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_idx = cv2.minMaxLoc(result)
        if max_score < max_val:
            max_score = max_val
            ls_exist  = [key, max_val, max_idx[0], max_idx[1]]

    if max_score > threshold[pattern]:
        ret = True
    else:
        ret = False
                
    return ret, ls_exist

# 数値を採取する関数
def getNumCpu(img, dic_templ, pattern):

    pixel     = {"ymdhm": 8,    "np": 8,    "score": 6,    "sp": 7}    # 数値の位の間にあるべき最小限のpixel幅
    maximum   = {"ymdhm": 12,   "np": 4,    "score": 2,    "sp": 2}    # 数値の桁数上限
    threshold = {"ymdhm": 0.7,  "np": 0.7,  "score": 0.7, "sp": 0.65}
    eps       = {"ymdhm": 0.4,  "np": 0.4,  "score": 0.4,  "sp": 0.4}

    dic_match_score = {}
    dic_match_range = []
    for key in dic_templ.keys():
        result  =  cv2.matchTemplate(img, dic_templ[key], cv2.TM_CCOEFF_NORMED)
        dic_match_score[key] = result       # ソース画像すべての領域のマッチングスコアを数字の数だけ保存
        _, max_val, _, _ = cv2.minMaxLoc(result) # 検証用
        match_y, match_x = np.where(result >= threshold[pattern])
        for x, y in zip(match_x, match_y):
            dic_match_range.append([x, y, dic_templ[key].shape[1], dic_templ[key].shape[0]])
        
        #print(key, "\n", dic_match_range)

    #print("dic_match_range:\n", dic_match_range)
    rectangles, weights = cv2.groupRectangles(dic_match_range, groupThreshold=2, eps=eps[pattern])
    len_digit = len(rectangles)
    if len_digit == 0:
        num = np.nan
        #print("    No values detected: ", rectangles)
        return num, rectangles
    else:
        sort_rec = rectangles[np.argsort(rectangles[:, 0])]

        digit = []*maximum[pattern]
        for x in sort_rec:
            match_score = 0
            max_score   = 0
            for key in dic_templ.keys():
                try:
                    match_score = dic_match_score[key][x[1]][x[0]]
                except Exception as e:
                    print(e, "\n",key, x[1], dic_match_score[key].shape, result.shape, sort_rec)
                if match_score > max_score:
                    max_score = match_score
                    max_digit = key
            digit.append(max_digit)

    #print("digit: ", digit, len(digit), type(digit))

    # patternごとの数値処理
    if pattern == "ymdhm":
        if   len_digit == 10:
            digit.insert(5, "0")
            digit.insert(4, "0")
            num = int("".join(digit))
        elif len_digit == 11:
            if abs(sort_rec[5][0] - (sort_rec[4][0]+sort_rec[4][2])) < pixel[pattern]:
                digit.insert(6, "0")
                num = int("".join(digit))
            else:
                digit.insert(4, "0")
                num = int("".join(digit))
        elif len_digit == 12:
            num = int("".join(digit))
        else:
            raise ValueError("あり得ない桁数です")
        
    if pattern == "np":
        digit.remove("p")
        if   digit[0] == "0":
            num = np.nan
        elif len(digit) > maximum["np"]:
            raise ValueError("\"p\"を除いたNPがあり得ない桁数です")
        else:
            num = int("".join(digit))
    
    if pattern == "score":
        if len_digit < 3:
            num = np.nan
        else:
            digit.remove("x")
            if  "n" in digit:
                num = np.nan
            elif   digit[0] == "0":
                digit.remove("0")
                num = int("".join(digit))
            else:
                num = int("".join(digit))

    if pattern == "sp":
        if len(digit) > maximum["sp"]:
            raise ValueError("あり得ない桁数です")
        else:
            num = int("".join(digit))
    
    return num, max_val #sort_rec

###################################################################################
# コマンドライン実行
if __name__ == "__main__":
    in_dir    = "../../data/sample_frame/result"
    templ_dir = "../../data/templates/num/num_battlelog_gry"

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

    templ_ymdhm = {}
    templ_np    = {}
    templ_score = {}
    templ_buki  = {}
    frame_trim  = []

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

    templ_img = {}
    for fname, path in dic_fname_path["ymdhm"].items():
        templ_ymdhm[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for fname, path in dic_fname_path["np"].items():
        templ_np[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for fname, path in dic_fname_path["score"].items():
        templ_score[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for fname, path in dic_fname_path["buki"].items():
        templ_buki[fname] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


    #"""
    # 単一ファイル用
    j = "6"
    src_gry = cv2.imread(os.path.join(in_dir, "01.png"),   cv2.IMREAD_GRAYSCALE)
    frame_trim = src_gry[trim[j][0]:trim[j][1], trim[j][2]:trim[j][3]]

    a, b = getExistCpu(frame_gpu_trim, templ_gpu_buki, pattern="buki")
    print("num: ", a, b)
    #print("sort_rec: \n", b)

    for x in b:
        cv2.rectangle(
            frame_trim, 
            (x[0], x[1]),
            (x[0]+x[2], x[1]+x[3]),
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4,
            shift=0
        )

    cv2.imshow("trim", frame_trim)
    cv2.waitKey()
    cv2.destroyAllWindows()

    """
    # フォルダ内全ファイル用
    files_file = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    for file in files_file:
        print(file)
        src_gry = cv2.imread(file,   cv2.IMREAD_GRAYSCALE)
        for i in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            frame_trim = src_gry[trim[i][0]:trim[i][1], trim[i][6]:trim[i][7]]
            frame_gpu_trim.upload(frame_trim)
            a1, b1 = getNum(frame_gpu_trim, templ_gpu_score, gpu_dst, pattern="score")
            frame_trim = src_gry[trim[i][0]:trim[i][1], trim[i][7]:trim[i][8]]
            frame_gpu_trim.upload(frame_trim)
            a2, b2 = getNum(frame_gpu_trim, templ_gpu_score, gpu_dst, pattern="score")
            frame_trim = src_gry[trim[i][0]:trim[i][1], trim[i][8]:trim[i][9]]
            frame_gpu_trim.upload(frame_trim)
            a3, b3 = getNum(frame_gpu_trim, templ_gpu_score, gpu_dst, pattern="score")
            print("{:>2}:{:>4}{:>4}{:>4}".format(i, a1, a2, a3))
    """