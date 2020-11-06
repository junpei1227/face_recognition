# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)

# HaarLike特徴抽出アルゴリズムのパス

# 任意のもののコメントアウトを外してください。
# HAAR_FILE = "haarcascade_frontalface_alt.xml"       #顔認識
#HAAR_FILE = "haarcascade_frontalface_alt2.xml"     #顔認識2
#HAAR_FILE = "haarcascade_frontalface_alt_tree.xml" #顔認識3
#HAAR_FILE = "haarcascade_frontalface_default.xml"  #顔認識4
#HAAR_FILE = "haarcascade_smile.xml"                #笑顔認識
#HAAR_FILE = "haarcascade_profileface.xml"          #顔（証明写真）認識

# HAAR_FILE = "haarcascade_eye.xml"                  #目認識
#HAAR_FILE = "haarcascade_lefteye_2splits.xml"      #左目認識
#HAAR_FILE = "haarcascade_righteye_2splits.xml"     #右目認識
#HAAR_FILE = "haarcascade_eye_tree_eyeglasses.xml"  #眼鏡認識

#HAAR_FILE = "haarcascade_fullbody.xml"             #全身認識
#HAAR_FILE = "haarcascade_upperbody.xml"            #上半身認識
#HAAR_FILE = "haarcascade_lowerbody.xml"            #下半身認識

#HAAR_FILE = "haarcascade_frontalcatface.xml"　     #猫の顔認識
#HAAR_FILE = "haarcascade_frontalcatface_extended.xml"　#猫の顔認識2
#HAAR_FILE = "haarcascade_licence_plate_rus_16stages.xml"#ロシアのナンバープレート認識
#HAAR_FILE = "haarcascade_russian_plate_number.xml"#ロシアのナンバープレートの数字認識

"""
上記のパスとなっているファイルはopencvをインストールした際にもう入っているようです。
下記の〇〇〇〇にユーザー名を入力し、××××にファイル名を入力して絶対パスで指定すると利用できます。
ただ、今回の訓練で作った環境でのパスなので、他環境だとまたパスが変わってしまいます。

"""

#絶対パス
#HAAR_FILE = "/home/〇〇〇〇/anaconda3/envs/py36/share/OpenCV/haarcascades/××××.xml"
# HAAR_FILE = "/home/tam/anaconda3/envs/py36/share/OpenCV/haarcascades/haarcascade_eye.xml"
HAAR_FILE = "/home/tam/anaconda3/envs/py36/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml"


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