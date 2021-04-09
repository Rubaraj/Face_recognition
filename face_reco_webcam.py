#Multi-Dimensional arrays and matrices, along with a large collection of high-level mathematical functions 
import numpy as np
#Library to identify face properties
import face_recognition as fr
#real-time access to computer hardware in this case camera --OPENCV
import cv2

#Library to use Email
import smtplib
import imghdr
from email.message import EmailMessage
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

#Mail send flag
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

        #Frame controls in video mode
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #Font style for Focus BOX
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #Send Email if there is a Unknown Person
        if name == "Unknown" and flag == "notsent":
            try:
                cv2.imwrite('images/intruder.png',frame)
                    
                #Framing the message
                msg = EmailMessage()
                msg['Subject'] = 'Intruder Alert !'
                #From address
                msg['From'] = EMAIL_ADDRESS                    
                #To address
                msg['To'] = 'rubarajankcs@hotmail.com'
                
                msg.set_content('Intruder Attached')
                with open('images/intruder.png','rb') as f:
                    file_data = f.read()
                    file_type = imghdr.what(f.name)
                    file_name = f.name

                msg.add_attachment(file_data,maintype = 'image',subtype=file_type, filename= file_name)
                
                #Login to Gmail
                with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
                    smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
                    smtp.send_message(msg)

                flag = "MailSent"
                print("Send Email")
                if os.path.exists("images/intruder.png"):
                    os.remove("images/intruder.png")
                else:
                    print("The file does not exist") 

            except Exception:
                print("Error: unable to send email")

    cv2.imshow('Webcam_facerecognition', frame)

    #Press q to quit the application Loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
