# -*- coding: utf-8 -*- 
import os
import sys
import glob
import numpy as np
from skimage import io
from sklearn import datasets
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


IMAGE_SIZE = 40
IMAGE_SIZE_Y = 40 
COLOR_BYTE = 3
CATEGORY_NUM = 4

## ラベル名(0～)を付けたディレクトリに分類されたイメージファイルを読み込む
## 入力パスはラベル名の上位のディレクトリ
def load_faceimage(path):

    # ファイル一覧を取得
    files = glob.glob(os.path.join(path, '*/*.png'))

    # イメージとラベル領域を確保
    images = np.ndarray((len(files), IMAGE_SIZE, IMAGE_SIZE_Y,
                            COLOR_BYTE), dtype = np.uint8)
    labels = np.ndarray(len(files), dtype=np.int)

    # イメージとラベルを読み込み
    for idx, file in enumerate(files):
       # イメージ読み込み
       image = io.imread(file)
       images[idx] = image

       # ディレクトリ名よりラベルを取得
       label = os.path.split(os.path.dirname(file))[-1]
       labels[idx] = int(label)

    # scikit-learn の他のデータセットの形式に合わせる
    flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE_Y * COLOR_BYTE))
    images = flat_data.view()
    return datasets.base.Bunch(data=flat_data,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)

#####################################
from sklearn import svm, metrics

## 学習データのディレクトリ、テストデータのディレクトリを指定する
if __name__ == '__main__':
    train_path = "train_data"
    test_path = "test_data"

    # 学習データの読み込み
    train = load_faceimage(train_path)

    # 手法:線形SVM
    classifier = svm.LinearSVC()

    # 学習
    classifier.fit(train.data, train.target)

    savefile = "face_result.pkl"
    # 学習モデルを保存する
    joblib.dump(classifier, savefile)
    print('学習結果はファイル {0} に保存されました'.format(savefile))


    # テストデータの読み込み
    test = load_faceimage(test_path)
    print(test.data.shape)

    # テスト
    # predicted = classifier.predict(test.data)
    loaded_cls = joblib.load(savefile)
    predicted = loaded_cls.predict(test.data)
    print(predicted)

    # 結果表示
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
