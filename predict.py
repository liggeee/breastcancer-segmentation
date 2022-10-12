import cv2
import numpy as np
from PIL import Image
from keras.utils import img_to_array
from keras.models import load_model

path_model = 'model/model_normal_malignant_mse_3-7_epoch_100_5e-5.h5'
# path = 'templates/images/1662436205.1745741.png'
def predict(path):
    data = np.zeros((1, 128, 128, 1))
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(128,128))
    pil_img = Image.fromarray(img)
    data[0]+=img_to_array(pil_img)
    data/=255.0
    model = load_model(path_model)
    y_pred = model.predict(data)
    Y= y_pred.reshape(y_pred.shape[0],128,128)
    cv = np.where(Y>0.5,1,0)
    pil_mask_img = Image.fromarray((cv[0] * 255).astype(np.uint8))
    blend_img = Image.blend(pil_img,pil_mask_img, 0.4)    
    return pil_img, blend_img

# data = np.zeros((1, 128, 128, 1))
# img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(128,128))
# pil_img = Image.fromarray(img)
# data[0]+=img_to_array(pil_img)
# nyoba = img_to_array(pil_img)
# print(nyoba.shape)
# data/=255.0
# model = load_model(path_model)
# y_pred = model.predict(data)
# Y= y_pred.reshape(y_pred.shape[0],128,128)
# cv = np.where(Y>0.5,1,0)
# pil_mask_img = Image.fromarray((cv[0] * 255).astype(np.uint8))
# blend_img = Image.blend(pil_img,pil_mask_img, 0.4)    
# return pil_img, blend_img