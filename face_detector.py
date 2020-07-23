import cv2
import sys
import glob
import numpy as np
from project import MACE_filter, teste_correlacao, teste_detection
import os
from matplotlib import pyplot as plt
import matplotlib.image as plt2 
import time

#variaveis de gatilho
result = 'negative'
user =  'none'
same =  'none'
aux =   0
aux_ =  0

cam = cv2.VideoCapture(0)

d = 48 

face_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

count = 0
init = time.time()
while(True):
    init = time.time()
    ret, img = cam.read()
    #Segmenta a imagem capturada em tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detecta a face do usuario na imagem e retorna suas medidas
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    #percorre as medidas de largura e altura provenientes de faces
    for (x,y,w,h) in faces:
        #desenha um retangulo na face do usuario
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        #recorta a face e atribui a imagem
        imagem = gray[y:y+h,x:x+w]
        #redimensiona para os padrões do projeto
        imagem = cv2.resize(imagem, (d,d))
        cv2.imshow('image', img)
        #realiza os testes de correlacao e toma o retorno da funcao
        output,msg, user = teste_detection(imagem)

        if output ==1: #and ((user != same) or (aux > (aux_+2))):
            #opcao de nomear a imagem de entrada e salvar em logs 
            #same = user
            #tempo = time.strftime("%d%m%H%M%S")
            #aux_ = int(time.strftime("%S"))
            #cv2.imwrite('image_data/input_logs/'+ tempo + '.jpg', gray[y:y+h,x:x+w])
            fim = time.time()
            tempo = fim - init
            print(msg)
            print(tempo)
            #time.sleep(2)
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
