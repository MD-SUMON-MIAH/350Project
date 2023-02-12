import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number={}
__class_number_to_name={}

__model=None

def classify_image(file_path):
    img=cv2.imread(file_path)
    scalled_img=cv2.resize(img,(32,32))
    img_har=w2d(img,'db1',7)
    scalled_img_har=cv2.resize(img_har,(32,32))
    combined_img = np.vstack((scalled_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    len_image_array=32*32*3+32*32
    #final=combined_img
    final=combined_img.reshape(1,len_image_array).astype(float)
    result=[]
    result.append(class_number_to_name(__model.predict(final)[0]))
    return result


def load_saved_artifacts():
    print("Loading saved artifacts....start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json","r") as f:
        __class_name_to_number=json.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items()}
    global __model
    if __model is None:
        with open("./artifacts/saved_model.pkl","rb") as f:
            __model=joblib.load(f)
    print("Loading saved artifacts...done")


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

if __name__ == '__main__':
    load_saved_artifacts()
    #print(classify_image(None, ))
    print(classify_image("./test_img/cyst.jpg"))
    print(classify_image("./test_img/miliria.jpg"))
    print(classify_image("./test_img/rings.jpg"))
    print(classify_image("./test_img/img_2.png"))

