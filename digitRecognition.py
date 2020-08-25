import cv2
import os
from keras.models import model_from_json
import numpy as np

path = r'C:\Users\Taha\Desktop\openCV\Images\\'

faceCasacade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
cap.set(10,200)

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")


while True:
    _, image = cap.read()
    # image = cv2.flip(image,1)
    cv2.imshow("Image",image)
    npimage = np.asarray(image)
    resized = cv2.resize(npimage,(28,28))
    gray_scale = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(gray_scale)
    image = image/255
    image = np.expand_dims(image, axis=-1)
    image = np.reshape(image,(1,28,28,1))
    result = model.predict_classes(image)
    print(str(result))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
