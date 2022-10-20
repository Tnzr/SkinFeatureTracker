import cv2
import dlib
import faceBlendCommon as face
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['figure.figsize'] = (8.0,8.0)
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'bilinear'


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# read the image
cap = cv2.VideoCapture(0)

def getLipsMask(size, lips):
    # Find Convex hull of all points
    hullIndex = cv2.convexHull(np.array(lips), returnPoints=False)
    # Convert hull index to list of points
    hullInt = []
    for hIndex in hullIndex:
        hullInt.append(lips[hIndex[0]])
    # Create mask such that convex hull is white
    mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))
    return mask

while True:
    start_time = time.time() # start time of the loop

    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    im = frame

    # Use detector to find landmarks


    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    points = face.getLandmarks(detector, predictor, imDlib)

    if not points:
        continue
    else:


        ## Select points of lips and create "lips mask"
        face = [points[x] for x in range(0, 28)]
        mouth = [points[x] for x in range(48, 60)]
        righteye = [points[x] for x in range(42, 48)]
        lefteye = [points[x] for x in range(36, 41)]
        clone_lips = frame.copy()
        for point in face:
            cv2.circle(clone_lips, point, 1, (0, 255, 0), -1)
        for point in mouth:
            cv2.circle(clone_lips, point, 1, (0, 0, 255), -1)
        for point in lefteye:
            cv2.circle(clone_lips, point, 1, (0, 0, 255), -1)
        for point in righteye:
            cv2.circle(clone_lips, point, 1, (0, 0, 255), -1)
        clone_lips = cv2.cvtColor(clone_lips, cv2.COLOR_BGR2RGB)
        contours = [np.asarray(face, dtype=np.int32)]
        (x, y, w, h) = cv2.boundingRect(contours[0])
        center = (int(x + w / 2), int(y + h / 2))
        mask = getLipsMask(im.shape, face)
        left_eye_mask = getLipsMask(im.shape, lefteye)
        right_eye_mask = getLipsMask(im.shape, righteye)
        mouth_mask = getLipsMask(im.shape, mouth)

        new_mask = mask - mouth_mask - right_eye_mask - left_eye_mask
        cv2.imshow("", new_mask)
        cv2.waitKey(0)

    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()