from __future__ import print_function
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
from PIL import ImageGrab
from grab_scr import grab_screen
import cv2
import numpy as np
from press_keys import PressKey,ReleaseKey, W, A, D, S
#from make_data import take_part


num_classes= 3

modelx = Sequential()

modelx.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(100,75,1)))
modelx.add(Activation('relu'))
modelx.add(Conv2D(32, (3, 3)))
modelx.add(Activation('relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
#modelx.add(Dropout(0.25))

modelx.add(Conv2D(64, (3, 3), padding='same'))
modelx.add(Activation('relu'))
modelx.add(Conv2D(64, (3, 3)))
modelx.add(Activation('relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
#modelx.add(Dropout(0.25))

modelx.add(Flatten())
modelx.add(Dense(512))
modelx.add(Activation('relu'))
#modelx.add(Dropout(0.5))
modelx.add(Dense(num_classes))
modelx.add(Activation('softmax'))


modelx.load_weights('my_model_weights_new_02.h5')


'''
WIDTH = 80
HEIGHT = 60


train_data = np.load('training_data_v3.npy')


test = train_data[:600]


x_test = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
y_test = np.array([i[1] for i in test])

y_test = keras.utils.to_categorical(y_test, num_classes)


res = modelx.predict(x_test)

res2 = []
for x in res:
    res2.append(np.argmax(x))
    

print(res2)
'''
 
def forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    
def fl():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D) 
    #time.sleep(0.1)
    #ReleaseKey(A)    
    
def fr():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A) 
    #time.sleep(0.1)    
    #ReleaseKey(D)    
'''
def fl():
    PressKey(W)    
    PressKey(A)
    
def fr():
    PressKey(W)
    PressKey(D)    
'''  

thr1 = 300
thr2 = 400
blur = 35
diliter= 10
eroditer = 5
color = (0.0,0.0,0.0)
  
def main():
    last_time = time.time()
    
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True): 
        screen0 = grab_screen(region=(300,450,650,540))
        vertices = np.array([[0,160],[0,120], [500,120], [500,160]], np.int32)

        l_gray = np.array([140,140,140])
        u_gray = np.array([150,150,150])

        screen0 = cv2.inRange(screen0, l_gray, u_gray)

        gray= screen0

        edges = gray
        #edges = cv2.Canny(gray, thr1, thr2)
        #cv2.imshow('img01', edges)
        edges = cv2.dilate(edges, None)
        #cv2.imshow('img02', edges)
        edges = cv2.erode(edges, None)
        #cv2.imshow('img03', edges)
        
        
        contur_bag = []
        _, conturs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contur in conturs:
            contur_bag.append([contur, cv2.isContourConvex(contur), cv2.contourArea(contur)])
        
        try:
            max_contur = max(contur_bag, key=lambda contur: contur[2])
        except ValueError:
            continue
        
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contur[0], (255))
        #cv2.imshow('imgx', mask)  
                
        mask = cv2.dilate(mask, None, iterations=diliter)
        mask = cv2.erode(mask, None, iterations= eroditer)
        mask = cv2.GaussianBlur(mask, (blur, blur), 2) 
        mask = cv2.resize(mask, (100,75))
        
        cv2.imshow('',mask)
        #moves = list(np.around(modelx.predict([screen.reshape(-1,100,75,1)])[0]))
        moves = (np.argmax(np.around(modelx.predict([mask.reshape(-1,100,75,1)])[0])))
        #print(np.argmax(np.around(modelx.predict([screen.reshape(-1,100,75,1)])[0])))
        #moves= moves[0]
        #print(moves)
        print((modelx.predict([mask.reshape(-1,100,75,1)])[0]))

        if moves == 0 and max(modelx.predict([mask.reshape(-1,100,75,1)])[0]) > 0.7:
            forward()

        elif moves == 1 and max(modelx.predict([mask.reshape(-1,100,75,1)])[0]) > 0.75:
            fl()

        elif moves == 2 and max(modelx.predict([mask.reshape(-1,100,75,1)])[0]) > 0.75:
            fr()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break       
        
main()
