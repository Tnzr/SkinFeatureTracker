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



webcam = cv.VideoCapture(0)
while True:
    start_time = time.time() # start time of the loop

    scc, img = webcam.read()
    img = cv.flip(img, 1)
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

    img = blank_img
    if results.multi_face_landmarks:
        if relative_x:
            for face in results.multi_face_landmarks:
                for n in range(len(face.landmark)-1):

                    start_point = (relative_x[n], relative_y[n])
                    end_point = (relative_x2[n], relative_y2[n])
                    color = (0, 255, 0)
                    thickness = 2
                    cv.line(img, start_point, end_point, color, thickness)

    cv.imshow('face mesh 3d', img)
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

    k = cv.waitKey(1)
    if k == ord('q'):
        break
webcam.release()
cv.destroyAllWindows()



