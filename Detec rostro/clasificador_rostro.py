import cv2
import os
from camera import getcamera  # Asegúrate de que este módulo exista

# Ruta a la carpeta que contiene las imágenes de entrenamiento
dataPath = './Data'  # Asegúrate que la carpeta se llame exactamente 'Data'
imagePaths = os.listdir(dataPath)
print('Imágenes detectadas:', imagePaths)

# Crear el modelo y cargar el entrenamiento
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo.xml')

# Cargar el clasificador de rostros
faceClassif = cv2.CascadeClassifier('rostros.xml')

# Inicializar la cámara
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        result = face_recognizer.predict(rostro)

        if result[1] < 75:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 25), 2, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo

    cv2.imshow('imagen', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cv2.destroyAllWindows()
cap.release()