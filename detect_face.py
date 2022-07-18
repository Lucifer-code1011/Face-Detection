import cv2 as cv

# Load the cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv.imshow('img', img)

    # Stop if escape key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
       break

#When everything done, release the capture
cap.release()
cv.destroyAllWindows()