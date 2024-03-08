import cv2
import numpy as np

# Load the DNN Face Detector model
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

def detect(image):
    
    # Get the height and width of the input image
    (h, w) = image.shape[:2]

    # Preprocess the image by resizing it and converting it to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Feed the blob as input to the DNN Face Detector model
    face_detector.setInput(blob)
    detections = face_detector.forward()

# Loop over the detections and draw a rectangle around each face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

    # Filter out weak detections
        if confidence > 0.5:
        # Get the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

        # Draw a rectangle around the face
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Show the output image
    cv2.imshow("Output", image)

# Open the default camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, display it
    if ret:
        #cv2.imshow('Camera Stream', frame)
        detect(frame)

        # Press 'q' to quit the application
        if cv2.waitKey(25) == 13:
            break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()