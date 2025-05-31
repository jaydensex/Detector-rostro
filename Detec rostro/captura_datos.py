# pip install imutils
import cv2
import os
import imutils
from camera import getcamera

print('Escribe tu nombre:')
personName = input()
dataPath = './data'
personPath = dataPath + '/' + personName

if os.path.exists(personPath):
    print('Persona ya registrada, sobreescribiendo datos...')
else:
    os.makedirs(personPath)
    print('Nueva persona, capturando datos...')

# abrir la cámara
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Cargar el detector de rostros
faceClassif = cv2.CascadeClassifier('rostros.xml')

contador = 0

while True:
    # Tomar fotografía
    ret, frame = cap.read()
    
    if not ret:
        break

    # Cambiando el tamaño de la foto
    frame = imutils.resize(frame, width=640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
        maxSize=(1000, 1000)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # copia de la imagen
    auxFrame = frame.copy()

    # obtenemos el recuadro del rostro
    rostro = auxFrame[y:y + h, x:x + w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

    # Guardar el rostro como imagen
    cv2.imwrite(personPath + '/rostro_{}.jpg'.format(contador), rostro)
    print('rostro_{}.jpg'.format(contador) + ' guardado')

    contador = contador + 1

    cv2.imshow('Mi camara', frame)

    if contador >= 10 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cierra camara y ventanas
cv2.destroyAllWindows()
cap.release()
