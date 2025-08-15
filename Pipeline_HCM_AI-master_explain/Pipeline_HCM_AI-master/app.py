
#IMPORT THU VIEN
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2 #--> Thu vien xu li hinh anh đọc, resize, chèn text, mã hóa JPEG
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # tat GPU --> Khong su dung GPU
import numpy as np
import pandas as pd
import glob 
import json # Model su dung Json lam database --> Chinh vi the chon cach xu li json
# --> Load file '.json' de load cac anh da xu li truoc do 


#Translation --> La ham dung de toi uu viec nhap vao query --> Tuc khi 
# nhap vao tieng anh/ tieng viet se cho ra ket qua tim kiem bang 2 ngon ngu
from utils.query_processing import Translation
from utils.faiss import Myfaiss

# http://0.0.0.0:5001/home?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')

app = Flask(__name__, template_folder='templates')

####### CONFIG #########
with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value 

LenDictPath = len(DictImagePath)
bin_file='faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")
# Myfaiss() --> Function nay dung de 
########################


# PHAN NAY THUC HIEN XU LI TREN WEB --> XUAT RA SO ANH DA TRUY XUAT DUOC 

@app.route('/home')
@app.route('/')
def thumbnailimg():
    print("load_iddoc")

    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100  #--> So anh trang truy xuat ra  
    
    # imgpath = request.args.get('imgpath') + "/"
    pagefile = []

    page_filelist = []
    list_idx = []

    if LenDictPath-1 > index+imgperindex:
        first_index = index * imgperindex
        last_index = index*imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1    
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1    

    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)


@app.route('/imgsearch') # xu li viec tim kiem hinh anh 
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/textsearch')  # tim kiem theo van ban 
def text_search():
    print("text search")

    pagefile = []
    text_query = request.args.get('textquery') # day la chuoi van ban nguoi dung nhap
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath') # chi ra duong dan day du cua hinh anh 
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280,720)) # set anh ve 1280x720

    # print(img.shape)
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA) # viet text len anh 

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)


#TONG QUAN CACH VAN HANH TRONG CODE NAY: 

#Input text search nhap vao --> Text qua ham translation() --> Chuyen bien ve dung
# ngon ngu roi moi bat dau --> Encode bang "ViT-B/32" --> Sau do moi truy van voi cac 
#hinh anh da xu li va dc luu truoc do,  dua vao "cosine-distance"
# 