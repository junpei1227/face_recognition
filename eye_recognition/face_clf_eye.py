# -*- coding: utf-8 -*-
import cv2
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from face_csv import get_csv_to_dic
 

def trimming_face_image(image, face, size=(40,40)):
    """
    画像から顔を切り取り,リサイズした画像を返す
    """
    for x, y, w, h in face:
        # スライシングで顔の部分を切り取る
        face_image = image[y:y+h, x:x+w]
        # リサイズする 
        face_image = cv2.resize(face_image, size)
        return face_image

def get_predicted(face_image, clsfile, IMAGE_SIZE=40, IMAGE_SIZE_Y=40, COLOR_BYTE=3):
    """
    """ 
    # 学習済のファイルを読み込む
    loaded_cls = joblib.load(clsfile)
    # 学習モデルの形式に変換
    flat_face_image = face_image.reshape((-1, IMAGE_SIZE * IMAGE_SIZE_Y * COLOR_BYTE))
    # 誰の目か予測する
    predicted = loaded_cls.predict(flat_face_image)[0]
    return predicted


# HaarLike特徴抽出アルゴリズムのパス
# 任意のパス
HAAR_FILE = "/home/tam/anaconda3/envs/py36/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml"
#  学習した分類機のファイル
clsfile = "face_result_eye_6.pkl"

IMAGE_SIZE = 40
IMAGE_SIZE_Y = 40
COLOR_BYTE = 3

font = cv2.FONT_HERSHEY_SIMPLEX

csv_file = "train_data_eye.csv"
name_dic = get_csv_to_dic(csv_file)

cap = cv2.VideoCapture(0)
while True: 
    # 映像データを読み込んでサイズ変更
    rst, stream = cap.read()
    stream = cv2.resize(stream, (320,240))
    
    # HaarLike特徴抽出アルゴリズムから分類器を作成
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    # 実際に分類を行う
    face = cascade.detectMultiScale(stream)

    # 認識したものが一つの時処理を行う
    if len(face) == 1:
        face_image = trimming_face_image(stream, face)
        predicted = get_predicted(face_image,clsfile)
        # 予測結果から名前を取得
        name_text = name_dic[str(predicted)]
        for x, y, w, h in face:
            # 認識した目を赤い四角で囲う
            cv2.rectangle(stream, (x,y), (x+w,y+h), (0,0,255), 1)
            # 名前を表示
            cv2.putText(stream, name_text,(x,y-10), font, 1, (0,255,0), 3, cv2.LINE_AA)

    # 画像をウインドウに表示
    cv2.imshow("img", stream)
    
    # 'q'を入力でアプリケーション終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
#終了処理
cap.release()
cv2.destroyAllWindows()