import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

import keras
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
import cv2
import imutils
import os
import time
from glob import glob

clas1 = [item[10:-1] for item in sorted(glob("./Dataset/*/"))]


from keras.preprocessing import image                  
from tqdm import tqdm


def path_to_tensor(img_path, width=224, height=224):
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)




class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 



from keras.models import load_model
model2 = load_model('trained_modelDNN1.h5')

from tkinter import filedialog
filename = filedialog.askopenfilename(title='open')

main_img = cv2.imread(filename)



image1= cv2.imread(filename) 

cv2.imshow('Original Image',image1)

hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
low_val = (0,100,0)
high_val = (179,255,255)

mask = cv2.inRange(hsv, low_val,high_val)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
result = cv2.bitwise_and(image1, image1,mask=mask)

cv2.imshow("Result", result)
cv2.imshow("Segmented", mask)

test_tensors = paths_to_tensor(filename)/255
pred=model2.predict(test_tensors)
pred=np.argmax(pred);
print('given Image Predicted  = '+clas1[pred])
