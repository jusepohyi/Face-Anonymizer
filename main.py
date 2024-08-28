from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def anonymize_face(image, blocks=3):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(0, len(ySteps)):
        for j in range(0, len(xSteps)):

            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)

    return image


print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 30, (400, 400), True)

anonymize = False
save = True

print()
print("Press 1 to Enable Anonymization")
print("Press 2 to Disable Anonymization")
print()
print("Press Q to Exit the Application")

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frame = cv2.flip(frame, 1)

    if anonymize:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                face = anonymize_face(face, 20)

                frame[startY:endY, startX:endX] = face

    vidout = cv2.resize(frame, (400, 400))
    writer.write(vidout)

    cv2.imshow("Face Anonymizer", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("1"):
        anonymize = True
    elif key == ord("2"):
        anonymize = False

writer.release()
cv2.destroyAllWindows()
vs.stop()
