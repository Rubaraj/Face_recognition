#Multi-Dimensional arrays and matrices, along with a large collection of high-level mathematical functions 
import numpy as np
#Library to identify face properties
import face_recognition as fr
#real-time access to computer hardware in this case camera
import cv2

#Library to use Email
import smtplib
#Library to get Environment variable
import os

#Retrive Data from Environment Variable from OS
EMAIL_ADDRESS = os.environ.get('EMAIL_USER');
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD');

#capture video from webcam to identify the person
video_capture = cv2.VideoCapture(0)

#Finding the known image from the path
Known_image = fr.load_image_file("Rubarajan.jpg")
Known_face_encoding = fr.face_encodings(Known_image)[0]

known_face_encondings = [Known_face_encoding]
known_face_names = ["rubarajan"]

flag = "notsent"

while True: 
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        print("Notification: " + name)

        #Send Email if there is a Unknown Person
        if name == "Unknown" and flag == "notsent":
            try:
                with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                    smtp.ehlo()
                    #Encrypting SMTP Traffic
                    smtp.starttls()
                    smtp.ehlo()
                    #Login to Gmail
                    smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
                    subject = 'Intruder Alert !'
                    body = 'Unknown Person in the Home'
                    #Framing the message
                    msg = f'Subject: {subject} \n\n {body}'
                    smtp.sendmail(EMAIL_ADDRESS,'rubarajankcs@hotmail.com',msg)
                flag = "MailSent"
                print("Send Email")
            except SMTPException:
                print("Error: unable to send email")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    #Press q to quit the application Loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()