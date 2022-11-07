import cv2
import dlib
import matplotlib.pyplot as plt
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    x = []
    y = []
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        for n in range(0, 68):
            x.append(landmarks.part(n).x)
            y.append(landmarks.part(n).y)

            # Draw a circle

    _2, frame2 = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame2, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces2 = detector(gray)
    x_2 = []
    y_2 = []
    for face in faces2:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        for n in range(0, 68):
            x_2.append(landmarks.part(n).x)
            y_2.append(landmarks.part(n).y)

            # Draw a circle

        if x:
            for n in range(0, 68):
                start_point = (x[n], y[n])
                end_point = (x_2[n], y_2[n])
                color = (0, 255, 0)
                thickness = 2
                cv2.line(frame2, start_point, end_point, color, thickness)

    cv2.imshow("Output", frame2)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()