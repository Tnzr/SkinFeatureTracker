import time
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
roi = []
sift = cv2.SIFT_create()
index_params = dict(algorithm=0, trees=5)
search_params = dict()

flann = cv2.FlannBasedMatcher(index_params, search_params)
orb = cv2.ORB_create()


def drawMatchDisplacement(dst, kp1, cl1, kp2, cl2, good):
    if cl1 is None:
        cl1 = (0, 0, 255)
    if cl2 is None:
        cl2 = (255, 0, 0)

    dst = cv2.drawKeypoints(dst, kp1, cl1)
    dst = cv2.drawKeypoints(dst, kp2, cl2)
    for match in good:
        kp1_idx, kp2_idx = match.queryIdx, match.trainIdx
        kp1_pos = np.array(kp1[kp1_idx].pt).astype(np.int)
        kp2_pos = np.array(kp2[kp2_idx].pt).astype(np.int)

        dst = cv2.line(dst, kp1_pos, kp2_pos, (0, 255, 0), 1)
    return dst

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    if len(roi) != 0:
        ft_dist = np.zeros(frame.shape, dtype="uint8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_edges = cv2.Canny(gray, 20, 50)

        # w, h = rgray.shape[::-1]
        t0 = time.time()
        kp1, des1 = sift.detectAndCompute(gray, None)
        t1 = time.time()
        print(f"SIFT runtime: {t1-t0}")
        # bf = cv2.BFMatcher()
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        ft_dist = drawMatchDisplacement(frame , kp1, None, kp2, None, good)
        img3 = cv2.drawMatches(frame, kp1, roi, kp2, good, None, flags=2)

        if len(good) > 4:
            query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = rgray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            # print(pts)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # print(matrix)
            # print(dst)
            # cv2.imshow("Homography", homography)
            # cv2.imshow("mask", mask)
            cv2.imshow("matches", img3)
            cv2.imshow("frame", frame)
            cv2.imshow("Feature Displacement", ft_dist)

    key = cv2.waitKey(1)
    if key == ord("s"):
        r = cv2.selectROI("select the area", frame)
        if r is not roi:
            roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            rgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # edges = cv2.Canny(rgray, 20, 50)
            cv2.imshow("roi", rgray)

            kp2, des2 = sift.detectAndCompute(rgray, None)

    if key == ord("q"):
        cap.release()
cv2.destroyAllWindows()
