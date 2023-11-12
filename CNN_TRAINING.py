import numpy as np
import pickle
import cv2

# Config:
width = 640
height = 480
threshold = 0.65

cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

pickle_input = open('trained_model_data.p', 'rb')
model = pickle.load(pickle_input)

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    flag, imgOriginal = cam.read()
    if not flag:
        print("Error Capturing Frames")
        break
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcess(img)
    img = img.reshape(1, 32, 32, 1)

    # predictions
    classIdx = int(model.predict_step(img))

    predictions = model.predict(img)
    probabilityVal = np.max(predictions)
    print(classIdx, probabilityVal)

    if probabilityVal > threshold:
        cv2.putText(imgOriginal, str(classIdx) + " " + str(probabilityVal), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
