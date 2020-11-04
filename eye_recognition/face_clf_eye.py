# -*- coding: utf-8 -*-
import cv2
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
 

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


# HaarLike特徴抽出アルゴリズムのパス
# 任意のパス
HAAR_FILE = "/home/tam/anaconda3/envs/py36/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml"
#  学習した分類機のファイル
clsfile = "face_result_eye.pkl"
# 読み込む
loaded_cls = joblib.load(clsfile)


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
        # 学習モデルの形式に変換
        face_image = trimming_face_image(stream, face)
        flat_face_image = face_image.reshape((-1, 40 * 40 * 3))
        # 誰の目か予測する
        predicted = loaded_cls.predict(flat_face_image)
        # 予測結果を表示する
        print(predicted)
    
    # 認識した目を赤い四角で囲う
    for x, y, w, h in face:
        cv2.rectangle(stream, (x,y), (x+w,y+h), (0,0,255), 1)

    # 画像をウインドウに表示
    cv2.imshow("img", stream)
    
    # 'q'を入力でアプリケーション終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
#終了処理
cap.release()
cv2.destroyAllWindows()