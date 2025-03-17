import cv2 as cv
import os
import numpy as np
import time

haar = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def match(ufolder, tface):
    '''used for matching the faces'''
    tface = tface.astype(np.float32)  #issue 1: type difference

    for img_name in os.listdir(ufolder):
        img_path = os.path.join(ufolder, img_name)
        sface = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        sface = cv.resize(sface, (200, 200)).astype(np.float32)

        diff = np.mean(np.abs(sface - tface))
        if diff<50:
            return True
    return False




def authenticate():
    '''take live photo and scan'''
    cap = cv.VideoCapture(0, cv.CAP_V4L)
    auth = False
    stime = time.time()

    while not auth:
        # issue 2: never ending time period for recognition
        elapsed_time = time.time() - stime 
        if elapsed_time > 10: 
            print("❌ Authentication Timed Out.")
            break


        ret, frame = cap.read()
        if not ret:
            break



        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resize = cv.resize(face,(200, 200))

            for user in os.listdir('dataset'):
                ufolder = os.path.join('dataset', user)
                if match(ufolder, face_resize):
                    print(f'Access Granted {user}!!')
                    auth = True
                    break
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.imshow("Authenticating...", frame)

            if cv.waitKey(1) == 27:  
                break

    cap.release()
    cv.destroyAllWindows()

    if not auth:
        print("❌ Authentication Failed. Face not recognized.")


if __name__ == "__main__":
    authenticate()

            