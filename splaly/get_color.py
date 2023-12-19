import os
#import sys

import cv2
import numpy as np   

# 画像を任意の手法でグレースケール化する関数
def cvtGry(img_color, pattern :str, binary=0, threshold=128):

    bina = {
                1 : cv2.THRESH_BINARY,       # threshold 以下の値を0、それ以外の値を maxValue にして2値化を行います。
                2 : cv2.THRESH_BINARY_INV,   # threshold 以下の値を maxValue、それ以外の値を0にして2値化を行います。
                3 : cv2.THRESH_TRUNC,        # threshold 以下の値はそのままで、それ以外の値を threshold にします。
                4 : cv2.THRESH_TOZERO,       # threshold 以下の値を0、それ以外の値はそのままにします。
                5 : cv2.THRESH_TOZERO_INV,   # threshold 以下の値はそのままで、それ以外の値を0にします。
                6 : cv2.THRESH_OTSU,         # 大津の手法で閾値を自動的に決める場合に指定します。
                7 : cv2.THRESH_TRIANGLE      # ライアングルアルゴリズムで閾値を自動的に決める場合に指定します。
           }

    # 後の計算のためにfloat64の浮動小数に変換
    img_color_f = img_color.astype(np.float64)
    # 画像のサイズを取得(height:高さ, width:幅, channel:チャンネル数)
    height, width, channel = img_color.shape[:3]

    # 出力画像
    result = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            blue = img_color_f[i, j, 0]
            green = img_color_f[i, j, 1]
            red = img_color_f[i, j, 2]

            if   pattern == "average":
                # 平均法
                result[i, j] = (red + green + blue) / 3
            elif pattern == "lightness":
                # ライトネス法
                result[i, j] = (max(blue, green, red) + min(blue, green, red)) / 2
            elif pattern == "luminosity":
                # ルミナンス法
                result[i, j] = 0.2126*red + 0.7152*green + 0.0722*blue

    if   binary != 0:
        # 2値画像化
            ret, result = cv2.threshold(result, maxval=255, thresh=threshold, type=bina[binary] )
        
    return result

# 画像内の指定された色の総面積と、座標（マスク画像）を返す関数
def getCrdColor():
    a=0


    ###################################################################################
# コマンドライン実行
if __name__ == "__main__":
    in_dir   = "C:/Users/ntkke/ProjectSplaly/data/images/buki/buki_battlelog"
    #in_path  = os.path.join(in_dir, sys.argv[1] + ".png")
    out_dir  = "C:/Users/ntkke/ProjectSplaly/data/images/buki/buki_battlelog_gry"
    #out_path = os.path.join(out_dir, sys.argv[1] + ".png")

    """
    img = cv2.imread(in_dir)
    img_binary = cvtGry(img, pattern="lightness", binary=0, threshold=100)
    #cv2.imwrite(os.path.join(out_dir, file), img_binary)
    cv2.imwrite(os.path.join(out_dir), img_binary)
    """
    
    files_file = [
        f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))
    ]
    """
    files = ("0000.png", "0008.png", "0022.png", "0057.png", "0061.png")
    idxs  = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240)
    """
    #for idx in idxs:
    for file in files_file:
        img = cv2.imread(os.path.join(in_dir, file))
        img_binary = cvtGry(img, pattern="lightness", binary=0, threshold=0)
        #cv2.imwrite(os.path.join(out_dir, file), img_binary)
        cv2.imwrite(os.path.join(out_dir, file), img_binary)
    
