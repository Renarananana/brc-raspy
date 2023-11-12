import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os
import threading

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


class EmotionDetector():
    def __init__(self) -> None:
        self.frame = None
        self.video = True
        self.faces = []
        self.prediction = None

        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        # If you want to train the same self.model or try other models, go for this
        # emotions will be displayed on your face from the webcam feed
        self.model.load_weights('model.h5')

        
    
    def start_video(self):
        # start the webcam feed
        self.cap = cv2.VideoCapture(0)
        t0 = time.time()
        self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.hola = False
        self.t = threading.Thread(target=self.emotion_detection, args=( ))
        self.t.start()
        while True:
            # Find haar cascade to draw bounding box around face
            ret, self.frame = self.cap.read()
            prev_t = t0
            t0 = time.time()
            delta_time = t0 - prev_t
            fps = 1/delta_time
            
            fps = str(int(fps))
            cv2.putText(self.frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            if not ret:
                break
            for (x, y, w, h) in self.faces:
                cv2.rectangle(self.frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                if self.prediction is not None:
                    maxindex = int(np.argmax(self.prediction))
                    print(self.prediction)
                    cv2.putText(self.frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            cv2.imshow('Video', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video = False
        self.t.join()
        self.cap.release()
        cv2.destroyAllWindows()




    def emotion_detection(self):
        while self.frame is None:
            time.sleep(0.1)
        while self.video:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in self.faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                self.prediction = self.model.predict(cropped_img, verbose = 0)

ed = EmotionDetector()
ed.start_video()
                
