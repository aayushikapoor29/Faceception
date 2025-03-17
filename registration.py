import cv2 as cv
import os

def register(name):
    folder_path = f"dataset/{name}"
    os.makedirs(folder_path, exist_ok=True)

    haar = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv.VideoCapture(0)
    count = 0

    while count<30:
        ret, frame = cap.read()
        # kuch capture nahi hua toh
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50,50))

        for (x,y,w,h) in faces:
            # extracring face
            face = gray[y:y+h, x:x+w]
            face_resize = cv.resize(face, (200, 200))

            cv.imwrite(f"{folder_path}/{count}.jpg", face_resize)
            count += 1

            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv.imshow("capturing face", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"registration complete")


name = input("Enter your name: ")

people = []
DIR = r'/home/aayushi/Faceception/dataset'
for i in os.listdir(DIR):
    people.append(i)

if name in people:
    print(f"you are already registered!!")
else:
    register(name)

