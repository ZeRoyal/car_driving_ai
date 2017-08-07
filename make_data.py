import numpy as np
from PIL import ImageGrab
import cv2
import time
from keys_util import key_check
from grab_scr import grab_screen
import os


thr1 = 300
thr2 = 400
blur = 35
diliter= 10
eroditer = 5
color = (0.0,0.0,0.0)

def grab_keys(keys):
    outs = {'W': [0], 'AW': [1], 'DW': [2]}
    k = outs[''.join(keys)]
    print(k)
    return outs[''.join(keys)]

f_name = 'train_01.npy'

if os.path.isfile(f_name):
    training_data = list(np.load(f_name))
else:
    training_data = []
        
def take_part(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):
        screen0 = grab_screen(region=(300,450,650,540))
        #screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
        #vertices = np.array([[0,160],[0,120], [500,120], [500,160]], np.int32)

        l_gray = np.array([140,140,140])
        u_gray = np.array([150,150,150])
        #screen0 = take_part(screen0, [vertices])
        screen0 = cv2.inRange(screen0, l_gray, u_gray)
        #cv2.imshow('img000', screen0)
        gray= screen0
        #gray = cv2.cvtColor(screen0,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img001', gray)
        #gray = cv2.inRange(gray, 110, 160)
        #cv2.imshow('img01', gray)

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
        cv2.imshow('imgx2', mask)  
        
        
        '''
        mask_stack_bgr = np.dstack([mask]*3)
        mask_stack_bgr  = mask_stack_bgr.astype('float32') / 255.0           
        img = screen0.astype('float32') / 255.0                 
        
        masked = (mask_stack_bgr * img) + ((1-mask_stack_bgr) * color) 
        masked = (masked * 255).astype('uint8')  
        cv2.imshow('img', masked) 
        '''        
        mask = cv2.resize(mask, (100,75))
        #screen=cv2.Canny(screen,200,300)
        #cv2.imshow('window1', screen)
        
        try:
            keys = key_check()
            out = grab_keys(keys)
            training_data.append([mask,out])
        except KeyError:
            pass
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        if len(training_data) % 200 == 0:
            #print(len(training_data))
            np.save(f_name,training_data) 
        
main()