import cv2
import numpy as np

class MyCudaStruct:
    def __init__(self):
        self.images    = {}
        self.loglist   = []
        self.shapedict = {}

    def set(self, key, image: np.ndarray):
        # 画像を辞書に保存
        self.images[key] = cv2.cuda_GpuMat(image)
        self.loglist.append(key)
        self.shapedict[key] = image.shape

    def get(self, key):
        # 辞書から画像を取得
        return self.images.get(key, None)
    
    def log(self):
        # keyのリストを取得
        return self.loglist

    def shape(self, key):
        # 辞書の画像サイズを取得
        return self.shapedict[key]

