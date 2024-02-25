import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from firebase_admin import credentials, storage, db
import firebase_admin
from datetime import datetime

class FaceAttendance:
    def __init__(self):
        self.cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(self.cred, {
            'databaseURL': 'https://face-attendance-realtime-930ed-default-rtdb.firebaseio.com/',
            'storageBucket': 'face-attendance-realtime-930ed.appspot.com'
        })

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.img_background = cv2.imread("resources/background.png")
        self.img_student = None
        self.image_mode_list = self.load_mode_images("resources/modes/")
        self.encode_list_known, self.student_ids = self.load_encodings("encodings.p")

        self.mode_type = 0
        self.counter = 0

    def load_mode_images(self, folder_path):
        mode_path = os.listdir(folder_path)
        return [cv2.imread(os.path.join(folder_path, path)) for path in mode_path]

    def load_encodings(self, file_path):
        with open(file_path, "rb") as file:
            encode_list_known_with_ids = pickle.load(file)
        return encode_list_known_with_ids

    def run(self):
        while True:
            student_info = {}
            success, img = self.cap.read()
            self.process_frame(img, student_info)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_frame(self, img, student_info):
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_current_frame = face_recognition.face_locations(img_small)
        encode_current_frame = face_recognition.face_encodings(img_small, face_current_frame)

        self.img_background[162:162+480, 55:55+640] = img
        self.img_background[44:44+633, 808:808+414] = self.image_mode_list[self.mode_type]

        if face_current_frame:
            for encode_face, face_loc in zip(encode_current_frame, face_current_frame):
                matches = face_recognition.compare_faces(self.encode_list_known, encode_face)
                face_distance = face_recognition.face_distance(self.encode_list_known, encode_face)

                if len(face_distance) != 0:
                    match_index = np.argmin(face_distance)

                if matches[match_index]:
                    name = self.student_ids[match_index]

                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    bbox = (x1+55, y1+162, x2-x1, y2-y1)
                    self.img_background = cvzone.cornerRect(self.img_background, bbox, 20, rt=0)

                    id = self.student_ids[match_index]
                    id = id.split("/")[1]

                    if self.counter == 0:
                        cvzone.putTextRect(self.img_background, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", self.img_background)
                        cv2.waitKey(1)
                        self.counter = 1
                        self.mode_type = 1

            if self.counter != 0:
                if self.counter == 1:
                # Get the student info from the database
                    student_info = db.reference("students").child(id).get()
                    print(student_info)
                    # Get the student image from the storage
                    bucket = storage.bucket()
                    blob = bucket.get_blob('images/' + student_info["image_filename"])

                    array = np.frombuffer(blob.download_as_bytes(), np.uint8)
                    self.img_student = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                    # Update data of attendance
                    datetime_object = datetime.strptime(student_info["last_attendance"], '%Y-%m-%dT%H:%M')

                    seconds_elapsed = (datetime.now() - datetime_object).total_seconds()
                    print(seconds_elapsed)
                    if seconds_elapsed > 60:
                        ref = db.reference("students").child(id)
                        student_info["total_attendance"] = int(student_info["total_attendance"]) + 1
                        ref.child("total_attendance").set(student_info["total_attendance"])
                        ref.child("last_attendance").set(datetime.now().strftime("%Y-%m-%dT%H:%M"))
                    else:
                        self.mode_type = 3
                        self.counter = 0
                        self.img_background[44:44+633, 808:808+414] = self.image_mode_list[self.mode_type]
                
            if self.mode_type != 3:


                if 10 < self.counter < 20:
                    self.mode_type = 2

                self.img_background[44:44+633, 808:808+414] = self.image_mode_list[self.mode_type]

                if self.counter <= 10:
                    student_info = db.reference("students").child(id).get()
                    cv2.putText(self.img_background, str(student_info["total_attendance"]), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    
                    cv2.putText(self.img_background, str(student_info["major"]), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.img_background, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.img_background, str(student_info["standing"]), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(self.img_background, str(student_info["year"]), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(self.img_background, str(student_info["starting_year"]), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)


                    (w, h), _ = cv2.getTextSize(str(student_info["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(self.img_background, str(student_info["name"]), (808 + offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    self.img_background[175:175+216, 909:909+216] = self.img_student

                self.counter += 1

                if self.counter >= 20:
                    self.counter = 0
                    self.mode_type = 0
                    student_info = []
                    self.img_student = []
                    self.img_background[44:44+633, 808:808+414] = self.image_mode_list[self.mode_type]

        else:
            self.counter = 0
            self.mode_type = 0
            self.img_background[44:44+633, 808:808+414] = self.image_mode_list[self.mode_type]

        cv2.imshow("Face Attendance", self.img_background)

if __name__ == "__main__":
    face_attendance = FaceAttendance()
    face_attendance.run()