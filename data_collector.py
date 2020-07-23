import cv2
import os
import time

cam = cv2.VideoCapture(0)   #atribuicao da classe videocapture ao objeto cam
cam.set(3, 640)             # largura da imagem
cam.set(4, 480)             # altura da imagem

#numero de imagens coletadas
img_tot = 100


face_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

#registro do nome do usuario
user_id = input('digite o nome do usuario: ') 

#criacao da pasta no banco de dados com o nome do usuario
if os.path.isdir('image_data/' + user_id) == False: os.mkdir('image_data/' + user_id)

count = 0
init = time.time()
while(True):
    #leitura da camera e armazenamento dos frames
    ret, img = cam.read()        
    #salva o frame em escala de cinza                           
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #deteccao da face na imagem 
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #desenha um retangulo na altura e largura da face detectada
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        #registro da imagem contida no retangulo
        cv2.imwrite("image_data/" + user_id + '/' + user_id + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    #encerra o laco quando atinge o numero pré-definido de imagens
    if count >= img_tot:                      
        print('Imagens coletadas com sucesso.')
        fim = time.time()
        tempo = fim-init
        print('o tempo que levou para coleta das imagens foi:', tempo)
        break
    #encerra o laco ao teclar a letra q
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
