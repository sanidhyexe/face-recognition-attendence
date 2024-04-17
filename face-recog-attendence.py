import face_recognition
import  numpy as np
import csv
import cv2
from datetime import datetime

video_capture = cv2.VideoCapture(1)

#load known faces

sanidhy_img =  face_recognition.load_image_file("faces/sanidhy.jpg")
sanidhy_encoding = face_recognition.face_encodings(sanidhy_img)[0]


avinash_img =  face_recognition.load_image_file("faces/avinash.jpg")
avinash_encoding = face_recognition.face_encodings(avinash_img)[0]

known_face_encodings = [sanidhy_encoding,avinash_encoding]
known_face_name = ["Sanidhy","Avinash"]
#list of expected students

students = known_face_name.copy()

face_locations = []
face_encoding = []

#get the current date and time

now = datetime.now()
current_date = now.strftime("%d-%m-%y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    recognise_faces = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_name[best_match_index]
        #add text if person is present
        if name in known_face_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomleftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,255,255)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " present ", bottomleftCornerOfText, font,
                        fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([current_time,name])

    cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()

cv2.destroyAllWindows()

f.close()

    
