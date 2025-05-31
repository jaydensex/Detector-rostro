import cv2
from camera import getcamera

camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier("rostros.xml")
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceClassif.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize= (120, 120),
                                        maxSize = (1000, 1000))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break