import cv2

# 顔認識のためのファイル
HAAR_FILE = "haarcascade_frontalface_alt.xml"
# 画像ファイルのパス
image_path = "Lenna.jpg"

# 画像を読み込む 
img = cv2.imread(image_path)
# HaarLike特徴抽出アルゴリズムから分類器を作成
cascade = cv2.CascadeClassifier(HAAR_FILE)
# 実際に分類を行う
face = cascade.detectMultiScale(img)

# 赤い四角で顔を囲う
for x, y, w, h in face:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)

# 画像の表示
cv2.imshow("img", img)

# 何かキーボードを押して終了
cv2.waitKey(0) 
cv2.destroyAllWindows()