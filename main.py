import cv2
import face_recognition
import os
import numpy as np
import time

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)  

KNOWN_DIR = "C:/Users/Kabir Azad/VisualStudi Python/Face_Detection_Attendance_System/Faces"
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_DIR, file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
            print(f"[INFO] Loaded: {file}")
        else:
            print(f"[WARNING] No face found in {file}")

frame_count = 0
process_every_n_frames = 5  
face_locations = []
face_names = []

while True:
    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()

    if frame_count % process_every_n_frames == 0:
        face_locations = face_recognition.face_locations(rgb_small, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "NOT FOUND"
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            face_names.append(name)

    frame_count += 1

   
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 3
        right *= 3
        bottom *= 3
        left *= 3
        color = (0, 255, 0) if name != "NOT FOUND" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
