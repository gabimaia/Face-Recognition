import cv2
import os
import numpy as np
import glob
from math import ceil
from matplotlib import pyplot as plt
import matplotlib.image as plt2 
import glob
from scipy.fftpack import fft

d = 48
safety_margin = 0.9
usuario = ''

def MACE_filter(path): #o parametro "d" serve para redimensionar as imagens de treinamento

    path = (path)                     #o valor padrao para d foi de 64
    N = len(path)
    soma = np.zeros((d**2, 1))
    X_ = np.zeros((d**2,N))
    i=0
    for img in path:
        n = cv2.imread(img,0)
        if n is None:
          pass
        else:
          n = cv2.resize(n,(d,d))
          X = np.fft.fft2(n)                      #fft da imagem                                                        
          X_[:,i] = X.flatten()                      #transforma a fft em um vetor d² e armazana na coluna de uma matrix X de dimensao d²xN
          soma = soma + np.abs(X.reshape(d**2,1))**2 #soma dos espectros de potencia
          i+=1

    media = soma/N                                 #Calcula a media dos espectros
    u = np.ones((N, 1))                            #Vetor coluna de N linhas.
    D = np.diag(media.flatten())                   #D = diagonal da média do espectro de potencia
    D_inv = np.linalg.pinv(D)                      #Inversa de D 

    #Calculando o filtro a partir dos parâmetros D, X e u especificados acima
    H = np.dot(np.dot(np.dot(D_inv,X_),np.linalg.pinv(np.dot(np.conj(np.transpose(X_)),np.dot(D_inv,X_)))),u)
    H = H.reshape((d,d)) 
    #H = D1*X*((X+*Di*X)^-1)*u
    return H

def teste_correlacao(img, filtro):
                                         #lendo a imagem de teste
    #half = int((d**2)/2)
    #img = cv2.imread(img,0)
    img = cv2.resize(img, (d,d))
                                         #calculando corrlacao
    img_fft = np.fft.fft2(img)           #transformada da imagem de teste
    teste = img_fft * np.conj(filtro)    #multiplica a transformada da imagem pelo conjugado do filtro
    teste = np.fft.ifft2(teste)          #calcula a inversa  
    teste = np.real(teste)*d*d           #normaliza os valores

    return teste

def teste_detection(imagem):
  
  path_filters = glob.glob('filtros/*.txt')
  
  for filters in path_filters:
    filtro = np.loadtxt(filters)
    teste = teste_correlacao(imagem, filtro)
    peak = np.max(teste)

    if peak >= safety_margin: 
      output = 1
      usuario = filters[8:-4]
      msg = ('usuario ' + usuario + ' detectado')
      break
    else: 
      output=0
      msg = ('usuario nao detectado')
      usuario = 'none'
      
  return output, msg, usuario

