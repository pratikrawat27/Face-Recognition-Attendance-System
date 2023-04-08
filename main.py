import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

CaptureVideo = cv2.VideoCapture(0)

# Load Known Faces
pratikImage = face_recognition.load_image_file("Faces/Pratik.jpg")
pratikEncoding = face_recognition.face_encodings(pratikImage)[0]

# Taking Encoding inside an array
KnownFaceE = [pratikEncoding]
KnownFaceN = ["Pratik"]

# List of Expected Students
students = KnownFaceN.copy()

FLocations = []
FEncodings = []

# Get The Current Date And Time
now = datetime.now()
current_date = datetime.strftime(now, "%Y-%m-%d")

# Open the CSV file for writing
with open(f"{current_date}.csv", "w+", newline="") as file:
    writer = csv.writer(file)

    while True:
        _, frame = CaptureVideo.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgbs = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize Faces
        face_locations = face_recognition.face_locations(rgbs)
        face_encodings = face_recognition.face_encodings(rgbs, face_locations)

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(KnownFaceE, face_encoding)
            face_distances = face_recognition.face_distance(KnownFaceE, face_encoding)
            best_match_index = np.argmin(face_distances)
            if match[best_match_index]:
                name = KnownFaceN[best_match_index]

                # Get the current time
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                # Write the attendance data to the CSV file
                writer.writerow([name, current_time])

                # Add Text if a person is present
                if name in KnownFaceN:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLC = (10, 100)
                    FontS = 1.5
                    FontC = (255, 0, 0)
                    thinkNess = 3
                    LineType = 2
                    cv2.putText(frame, name + ' Present', bottomLC, font, FontS, FontC, thinkNess, LineType)

                    if name in students:
                        students.remove(name)
                        writer.writerow([name, current_time])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('e '):
            break

CaptureVideo.release()
cv2.destroyAllWindows()

# Close the CSV file
file.close()