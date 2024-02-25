from flask import Flask, render_template, request
import os
import firebase_admin
from firebase_admin import credentials, storage, db
import face_recognition
import numpy as np
import pickle
import cv2
import io


# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://face-attendance-realtime-930ed-default-rtdb.firebaseio.com/',
    'storageBucket': 'face-attendance-realtime-930ed.appspot.com'
})

# Create a reference to the 'students' node in the database
ref = db.reference('students')


app = Flask(__name__)


# Crop images to only include the face and save them to a new file 216 by 216

def cropped_face(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations
    face_locations = face_recognition.face_locations(image_rgb)

    # If a face is detected
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_height = bottom - top
        face_width = right - left

        # Calculate padding (extra space) around the face
        padding_height = max(0, ((216 - face_height) // 2))
        padding_width = max(0, ((216 - face_width) // 2))

        # Calculate the coordinates of the cropped image
        top = max(0, top - padding_height)
        bottom = min(image_rgb.shape[0], bottom + padding_height)
        left = max(0, left - padding_width)
        right = min(image_rgb.shape[1], right + padding_width)

        # Crop the image
        cropped_image = image_rgb[top:bottom, left:right]

        # Resize the cropped image to 216x216
        cropped_image = cv2.resize(cropped_image, (216, 216))

        # Convert the cropped image back to BGR color space
        cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # Convert the cropped image back to a file
        is_success, im_buf_arr = cv2.imencode(".jpg", cropped_image_bgr)
        byte_im = im_buf_arr.tobytes()
        image = io.BytesIO(byte_im)

        return image
    return None


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the form submission
@app.route('/add_student', methods=['POST'])
def add_student():
    # Get the form data
    name = request.form['name']
    major = request.form['major']
    starting_year = request.form['starting_year']
    total_attendance = request.form['total_attendance']
    standing = request.form['standing']
    year = request.form['year']
    last_attendance = request.form['last_attendance']
    image = request.files['image']

    # Generate a unique ID for the student
    student_id = str(hash(name + major + starting_year))

    # Rename the uploaded image to the student's ID
    image_filename = student_id + '.' + image.filename.split('.')[-1]
    
    

    # Upload the image to Firebase Storage
    image = cropped_face(cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR))
    image_path = os.path.join('static/images', image_filename)
    # Save the image to a file
    with open(image_path, 'wb') as f:
        f.write(image.getbuffer())

    if image:
        bucket = storage.bucket()
        blob = bucket.blob('images/' + image_filename)
        blob.upload_from_file(image)

        # Seek back to the start of the file
        image.seek(0)

        image_list = []
        student_ids = []
        blobs = bucket.list_blobs()

        for blob in blobs:
            # Download the blob to a file-like object in memory
            blob_bytes = blob.download_as_bytes()
            image_np = np.fromstring(blob_bytes, np.uint8)
            image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_list.append(image_rgb)
            student_ids.append(blob.name.split(".")[0])  # Get the student id from the file name

        def find_encodings(images):
            encode_list = []
            for img in images:
                encode = face_recognition.face_encodings(img)[0]
                encode_list.append(encode)
            return encode_list

        print("Encoding Images.....")
        encode_list_known = find_encodings(image_list)
        encode_list_known_with_ids = [encode_list_known, student_ids]
        print("Encoding Complete")

        with open("encodings.p", "wb") as file:
            pickle.dump(encode_list_known_with_ids, file)
        print("Encodings saved to encodings.p")

            

        # Save the student data to the database
        student_data = {
            "name": name,
            "major": major,
            "starting_year": int(starting_year),
            "total_attendance": int(total_attendance),
            "standing": standing,
            "year": int(year),
            "last_attendance": last_attendance,
            "image_filename": image_filename
        }

        ref.child(student_id).set(student_data)

        return 'Student added successfully!'
    
    return 'No face detected!'

if __name__ == '__main__':
    app.run(debug=True)
