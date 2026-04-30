import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not opening!")
    exit()

student_id = input("Enter Student Roll No: ")

path = os.path.join("dataset", student_id)
os.makedirs(path, exist_ok=True)

count = 0
last_capture_time = 0

print("Move your head slowly...")

while True:

    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:

        current_time = time.time()

        # faster capture
        if current_time - last_capture_time > 0.1:

            count += 1

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200,200))

            cv2.imwrite(os.path.join(path, f"{count}.jpg"), face_img)

            last_capture_time = current_time

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(frame,
                f"Captured: {count}/30",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Face collection completed.")