import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cv2
import glob
import os
import time

def main():
    '''
    <説明>
    ・このプログラムは，Deep learning用に画像のデータセットを作成します．
    ・スキャン画像と２色トレース，３色トレースの画像をそれぞれ対応したランダムな場所でトリミングします．
    ・１試料の画像サンプル(scan,2trace,3trace)につき各500枚ずつ，512×512のカラー画像を作成します．
    ・最終的には"productions"というフォルダ(名前変更可)が作成され，その中にスキャン画像，２色，３色の
    トリミング画像がそれぞれのフォルダ内に格納されます．
    ・テスト用の画像を作成するならば，46行目の指示に従ってください．(テスト用の画像はトレーニングで使わない画像であること)

    '''
    t1 = time.time()
    s_datas = glob.glob('*_scan.tif')    #scan画像は，'サンプル名_scan.tif'で保存しておく
    t2_datas = glob.glob('*_trace2.tif') #2trace画像は，'サンプル名_trace2.tif'で保存しておく
    t3_datas = glob.glob('*_trace3.tif') #3trace画像は，'サンプル名_trace3.tif'で保存しておく

    img_set = Parallel(n_jobs=-1)(delayed(imread)(a) for a in zip(s_datas, t2_datas, t3_datas))
    #3種類で1セットの画像読み込み

    num_figures = int((len(img_set)) * 500)
    #得られる画像datasetの総枚数
    # len(img_set) = 画像のセット数
    '''
    fig_num = num_figures // 500
    image = img_set[fig_num]
    print(fig_num)
    print(image)
    print(type(image))

    cut_images = imcut(image)
    print(cut_images[0])
    print(cut_images[2].shape)
    '''
    make_new_dirs("productions")
    #保存先フォルダの作成　("フォルダ名") --> productions\original, 2traced, 3traced のフォルダを作成
    #既にフォルダが作成されていれば無視して上書きされる

    # make_new_dirs("test_set")
    #テスト用の画像を作成するときは，42行目をコメントアウトして，46行目の "# "を消してください．

    productions = Parallel(n_jobs=-1, backend='threading')(delayed(fiv_hund_images)(b, img_set) for b in range(num_figures))
    #画像のセット数 × 500枚のランダムトリミング画像の生成
    t2 = time.time()

    elapsed_time = t2 - t1
    print("Number of Image-data-set: ",len(productions))
    print("Elapsed time: ",elapsed_time)

    '''
    print(img_set[0][0]) #0番目の画像のスキャン画像[0] (scan[0], trace2[1], trace3[2])
    print(type(img_set[0][0])) #Output--> <class 'numpy.ndarray'>
    print(img_set[0][0].shape) #(高さ，幅，色)　AN02-1の例：(14680, 1151, 3)
    '''


#Process
def imread(a):

    s_img = cv2.imread(a[0]) #scan画像
    t2_img = cv2.imread(a[1]) #なぞった画像
    t3_img = cv2.imread(a[2])

    return s_img, t2_img, t3_img

def imcut(img):

    sq = 512
    y_max = img[0].shape[0] - sq
    x_max = img[0].shape[1] - sq
    #512×512の画像を500枚作成する
    #0～最大値*のランダム整数を1個生成
    x = np.random.randint(0,x_max)
    y = np.random.randint(0,y_max)
    
    #ある座標（x,y）が定まった
    #print(x,y)

    #ある座標（x,y）から切り出したいサイズの端を指定（+sq）
    x_lim = x + sq
    y_lim = y + sq

    #print(x_lim,y_lim)

    #トリミング作業
    #トリミング画像=元画像[高さ範囲，幅範囲]
    cut_simg = img[0][y:y_lim,x:x_lim]
    cut_t2img = img[1][y:y_lim,x:x_lim]
    cut_t3img = img[2][y:y_lim,x:x_lim]

    return cut_simg, cut_t2img, cut_t3img


def fiv_hund_images(b, img_set):

    fig_num = b // 500
    image = img_set[fig_num]
    cut_images = imcut(image)

    #トリミングされた画像の保存名を定義（folder name）
    #i番目という情報を入れる
    #保存するフォルダ（production/）を指定

    '''
    fn1 = "Cut_simg" + str(b) + ".tif" #scanのトリミング画像
    fn2 = "Cut_t2img" + str(b) + ".tif" #trace2のトリミング画像
    fn3 = "Cut_t3img" + str(b) + ".tif" #trace3のトリミング画像
    '''

    fn1 = "productions/original/Cut_simg" + str(b) + ".tif" #scanのトリミング画像
    fn2 = "productions/2traced/Cut_t2img" + str(b) + ".tif" #trace2のトリミング画像
    fn3 = "productions/3traced/Cut_t3img" + str(b) + ".tif" #trace3のトリミング画像


    #トリミングされた画像を保存（保存名，保存したい画像）
    cv2.imwrite(fn1,cut_images[0])
    cv2.imwrite(fn2,cut_images[1]) 
    cv2.imwrite(fn3,cut_images[2])

    return cut_images


def make_new_dirs(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        scan = path + "\\original"
        t2race = path + "\\2traced"
        t3race = path + "\\3traced"
        os.makedirs(scan)
        os.makedirs(t2race)
        os.makedirs(t3race)


if __name__ == '__main__':
    main()

