import cv2
import numpy as np
import keras
from keras.models import load_model


capture=cv2.VideoCapture(0)


dictionary={0: 'anger',
            1: 'anger',
            2: 'fear',
            3: 'happiness',
            4: 'sadness',
            5: 'surprise',
            6: 'neutral'} ;

colors={0:(0,0,255),1:(0,0,255),2:(0,0,255),3:(0,255,0),4:(0,0,255),5:(100,100,100),6:(255,0,0)}


from keras.models import load_model

model=load_model('improved.h5');


face=cv2.CascadeClassifier('/Users/raunakjohar/Desktop/haarcascade_frontalface_default.xml')

font=cv2.FONT_HERSHEY_SIMPLEX


while(True):
    ret, frame = capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    frame=cv2.flip(frame,180)

    faces=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE);

    for (x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 1 )
      cv2.putText(frame, 'FACE DETECTED', (x-5, y-5), font,0.7, (0, 255, 0), 2, cv2.LINE_AA);

      roi=gray[y:y+h,x:x+w];
      roi=cv2.resize(roi,dsize=(48,48))
      roi=np.array(roi)
      roi = (roi / 255);
      size=(48,48)
      roi=roi.reshape(48,48)

      roi=roi[np.newaxis,:,:,np.newaxis];
      output=model.predict(roi);
      key=np.argmax(output);

      cv2.putText(frame,dictionary[key], (50,50),font ,1,colors[key],2, cv2.LINE_AA);



    cv2.imshow('EMOTION',frame);

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break;

capture.release();
cv2.destroyAllWindows();
