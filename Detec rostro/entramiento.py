import cv2
import os
import numpy as np

dataPath = './data'
listaPersonas = os.listdir(dataPath)

print("Lista de personas", listaPersonas)

labels = []
facesData = []
label = 0

print("Leyendo las imagenes....")
for nameDir in listaPersonas:
    #ruta para cada carpeta que esta dentro de data
    personPath = dataPath + '/' + nameDir
    
    for fileName in os.listdir(personPath):
        print("Rostros: ", nameDir + '/' + fileName)
        
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        
    label = label + 1
    
    
faces_recognizer = cv2.face.LBPHFaceRecognizer_create()

#entrenando para reconocer rostros
print("Entrenando modelo-...")
faces_recognizer.train(facesData, np.array(labels))

#almacenar el modelo obtenido
faces_recognizer.write("modelo.xml")
print("Modelo almacenado")