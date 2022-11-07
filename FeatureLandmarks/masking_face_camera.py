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


def getMask(size, lips):
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
    im = frame.copy()

    # Use detector to find landmarks
    imDlib = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    points = face.getLandmarks(detector, predictor, imDlib)

    if not points:
        continue
    ## Select point and create "mask"

    face2 = 0
    mouth = 0
    righteye = 0
    lefteye= 0
    face2 = [points[x] for x in range(0, 28)]
    mouth = [points[x] for x in range(48, 60)]
    righteye = [points[x] for x in range(42, 48)]
    lefteye = [points[x] for x in range(36, 41)]
    mask = getMask(im.shape, face2)
    left_eye_mask = getMask(im.shape, lefteye)
    right_eye_mask = getMask(im.shape, righteye)
    mouth_mask = getMask(im.shape, mouth)

    new_mask = mask - mouth_mask - right_eye_mask - left_eye_mask
    cv2.imshow("FACE MASK", new_mask)
    fps = (time.time() - start_time)**-1
    print("FPS: ", fps) # FPS = 1 / time to process loop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()