import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('C:\code\emotion detection final\haarcascade_frontalface_default.xml')

def captureVideo():
    return cv2.VideoCapture(0)

def detectFace(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray,1.1,4)

def detectEmotion(frame):
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )
    emotion = result["dominant_emotion"]
    return str(emotion)

def showResult(frame,txt):
    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)



cap = captureVideo()

while True:
    ret,frame = cap.read()

    faces = detectFace(frame)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    txt = detectEmotion(frame)

    showResult(frame,txt)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()