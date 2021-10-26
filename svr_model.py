import pickle
from flask import Flask, render_template, request
import os
from random import random
import cv2
import numpy as np
import sys
from models.model_celeb.opensetmodel import openSetClassifier,predict_person_from_image
import torch
from models.MBF_glin360.MBF_glin import *
from models.face_detector.BlazeDetector import BlazeFaceDetector
import face_alignment
from models.face_align.face_align_model import get_dst, alignment
import json


# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

#detector = MTCNN()
detector = BlazeFaceDetector(typ="front")
fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

embedding = MobileFaceNet(512,7,7)

ckpt_embed = torch.load("models/MBF_glin360/Backbone_glint360k_step3.pth",map_location='cpu')
embedding = load_state_dict_MB(embedding,ckpt_embed)
embedding.eval()

ckpt_facereg = torch.load("models/model_celeb/FAR_1percent_90DIR_0.1percent_80DIR_best.pth",map_location='cpu')
model_celeb = openSetClassifier(num_classes=103)
model_celeb.to('cpu')
net_dict = model_celeb.state_dict()
pretrained_dict = {k: v for k, v in ckpt_facereg['net'].items() if k in net_dict}
if 'anchors' not in pretrained_dict.keys():
    pretrained_dict['anchors'] = ckpt_facereg['net']['means']
model_celeb.load_state_dict(pretrained_dict)
anchor_mean = pickle.loads(open('models/model_celeb/anchor_mean.pickle', "rb").read())
model_celeb.set_anchors(torch.Tensor(anchor_mean))
model_celeb.eval()

dist_label_name = json.loads(open('dist_label_vnceleb.txt', encoding='utf-8').read())


# Hàm xử lý request
@app.route("/check_list")
def check_list():
    with open("dist_label_vnceleb.txt","r",encoding="utf-8") as f:
        content = f.read()
    return content

@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                #print(image.filename)
                #print(app.config['UPLOAD_FOLDER'])
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                #print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                faces_locations = detector(frame, 1)
                if len(faces_locations) != 0:
                    (x1, y1, width, height) = faces_locations[0]
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    face = frame[y1:y2, x1:x2]
                    d = abs(width - height)//2
                    if width >= height:
                        y2 = y1 - d + width
                        face = frame[y1-d:y2, x1:x2]
                    else:
                        x2 = x1 - d + height
                        face = frame[y1:y2,x1-d:x2]
                    face_rgb = face[...,::-1]
                    landmarks = fa_model.get_landmarks(face_rgb)
                    dst = get_dst(landmarks)
                    face_112x112 = alignment(face, dst, 112, 112)
                    face_fn = preprocess_data_MB(face_112x112)
                    embed_result = embedding(face_fn)
                    th = 3.600184650310259
                    #th = 999
                    person,score = predict_person_from_image(embed_result,model_celeb,th)
                    fn_person = dist_label_name[str(person)] if person != -999 else "Unknown"
                    if fn_person == "":
                        fn_person = str(person)
                    extra = f"Dự đoán : {fn_person}"

                    cv2.imwrite(path_to_save, face)

                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", idBoolean = True, extra=extra)
                else:
                    return render_template('index.html', msg='Không nhận diện được khuôn mặt')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được khuôn mặt')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
