import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number={}
__class_number_to_name={}

__model=None

def classify_image(image_base64_string,file_path=None):
    #img=cv2.imread(file_path)
    img=get_cv2_image_from_base64_string(image_base64_string)

    scalled_img=cv2.resize(img,(50,50))
    img_har=w2d(img,'db1',7)
    scalled_img_har=cv2.resize(img_har,(50,50))
    combined_img = np.vstack((scalled_img.reshape(50*50*3,1),scalled_img_har.reshape(50*50,1)))
    len_image_array=50*50*3+50*50
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


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img




def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_b64_test_image_for_cyst():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    #print(classify_image(None, ))
    #print(classify_image("./test_img/cyst.jpg"))
    #print(classify_image("./test_img/miliria.jpg"))
    #print(classify_image("./test_img/rings.jpg"))
    #print(classify_image("./test_img/img_2.png"))
    #print(classify_image(get_b64_test_image_for_cyst(),None))
    #print("sumon")

