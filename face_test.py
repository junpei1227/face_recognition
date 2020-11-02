# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)

# HaarLike特徴抽出アルゴリズムのパス
# 任意のパス
HAAR_FILE = "haarcascade_frontalface_alt.xml"

while True: 
    # 映像データを読み込んでサイズ変更
    rst, stream = cap.read()
    stream = cv2.resize(stream, (320,240))
    
    # HaarLike特徴抽出アルゴリズムから分類器を作成
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    # 実際に分類を行う
    face = cascade.detectMultiScale(stream)
    
    # 認識した顔を赤い四角で囲う
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