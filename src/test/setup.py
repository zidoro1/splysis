import os
import sys
import cv2


# １秒ごとに動画のframeを取り出す関数
def save_frames(in_path: str, out_dir: str, ext="png"):

    # 動画の読み込み
    cap = cv2.VideoCapture(in_path)
    idx = 0
    print("width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps   :", cap.get(cv2.CAP_PROP_FPS))
    print("f_all :", cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            idx += 1

            # 0秒のフレームを保存。CAP_PROP_POS_FRAMESは現在のフレーム
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
                out_path = os.path.join(out_dir, "{}.{}".format("0000", ext))
                cv2.imwrite(out_path, frame)

                """
                # 60(f)以下の場合処理続行。
                elif idx < cap.get(cv2.CAP_PROP_FPS):
                    continue

                # 60(f)ごとにフレームを保存
                else:
                    second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                    filled_second = str(second).zfill(4)
                    out_path = os.path.join(out_dir, "{}.{}".format(filled_second, ext))
                    cv2.imwrite(out_path, frame)
                    idx = 0
                """
            else:
                second = idx
                filled_second = str(second).zfill(4)
                out_path = os.path.join(out_dir, "{}.{}".format(filled_second, ext))
                cv2.imwrite(out_path, frame)
        else:
            break
        
    cap.release()

###################################################################################
# コマンドライン実行
if __name__ == "__main__":
    in_dir    = "../../data/sample_video"
    in_path   = os.path.join(in_dir, sys.argv[1])      # 動画ファイル名 
    out_dir   = "../../data/sample_video/frame"
    out_path  = os.path.join(out_dir)  # sample_asariみたいなフォルダ名
    save_frames(in_path, out_path, "png")