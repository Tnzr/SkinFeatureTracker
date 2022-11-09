# Code with comments
import cv2 as cv
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import time
import inspect
mpfacemesh = mp.solutions.face_mesh
FaceMesh = mpfacemesh.FaceMesh(max_num_faces=1)
mpdraw = mp.solutions.drawing_utils
drawspec1 = mpdraw.DrawingSpec(color=(255, 255, 0), circle_radius=0, thickness=1)
drawspec2 = mpdraw.DrawingSpec(color=(0, 255, 0), circle_radius=0, thickness=1)
img2 = None
background = False

webcam = cv.VideoCapture(0)
while True:
    start_time = time.time() # start time of the loop
    scc, img = webcam.read()

    img = cv.flip(img, 1)
    bg_img = img if background else np.zeros(img.shape)

    h, w, c = img.shape
    blank_img = np.zeros((h, w, c), np.uint8)
    results = FaceMesh.process(img)
    relative_x = []
    relative_y = []
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                shape = img.shape
                relative_x.append(int(x * shape[1]))
                relative_y.append(int(y * shape[0]))
                cv.circle(bg_img, (int(x * shape[1]), int(y * shape[0])), 4, (255, 0, 0), -1)

    scc, img2 = webcam.read()
    img2 = cv.flip(img2, 1)
    h, w, c = img2.shape
    blank_img = np.zeros((h, w, c), np.uint8)
    results = FaceMesh.process(img2)
    relative_x2 = []
    relative_y2 = []
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                shape = img.shape
                relative_x2.append(int(x * shape[1]))
                relative_y2.append(int(y * shape[0]))
                cv.circle(bg_img, (int(x * shape[1]), int(y * shape[0])), 3, (0, 0, 255), -1)


    if results.multi_face_landmarks:
        if relative_x:
            for face in results.multi_face_landmarks:
                for n in range(len(face.landmark)-1):

                    start_point = (relative_x[n], relative_y[n])
                    end_point = (relative_x2[n], relative_y2[n])
                    color = (0, 255, 0)
                    thickness = 1
                    cv.line(bg_img, start_point, end_point, color, thickness)


    cv.imshow('face mesh 3d', bg_img)
    print("FPS: ", 1.0 / (time.time() - start_time), "B: ", int(background)) # FPS = 1 / time to process loop
    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord("b"):
        background = not background
webcam.release()
cv.destroyAllWindows()



