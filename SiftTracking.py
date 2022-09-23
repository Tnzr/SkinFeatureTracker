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
kp1, des1 = [], None
kp2, des2 = [], None
prev_frame = None


def drawMatchDisplacement(dst, kp1, cl1, kp2, cl2, good, only_matches=True, ):
    if cl1 is None:
        cl1 = (0, 0, 255)
    if cl2 is None:
        cl2 = (255, 0, 0)
    kp1_idx = []
    kp2_idx = []

    for match in good:
        kp1_id, kp2_id = match.queryIdx, match.trainIdx
        kp1_pos = np.array(kp1[kp1_id].pt).astype(np.int)
        kp2_pos = np.array(kp2[kp2_id].pt).astype(np.int)
        if np.sqrt(np.mean((kp1_pos-kp2_pos)**2)) > 5:
            dst = cv2.line(dst, kp1_pos, kp2_pos, (0, 255, 0), 2)
            if only_matches:
                kp1_idx.append(kp1_id)
                kp2_idx.append(kp2_id)
    if only_matches:
        match_kp1 = [kp1[match_idx] for match_idx in kp1_idx]
        match_kp2 = [kp2[match_idx] for match_idx in kp2_idx]
        dst = cv2.drawKeypoints(dst, match_kp1, cl1)
        dst = cv2.drawKeypoints(dst, match_kp2, cl2)
    else:
        dst = cv2.drawKeypoints(dst, kp1, cl1)
        dst = cv2.drawKeypoints(dst, kp2, cl2)
    return dst


while cap.isOpened():
    t0 = time.time()
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray, None)
    t1 = time.time()
    if len(kp2) != 0:
        ft_dist = np.zeros(frame.shape, dtype="uint8")
        # frame_edges = cv2.Canny(gray, 20, 50)

        # w, h = rgray.shape[::-1]

        # bf = cv2.BFMatcher()
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)

        ft_dist = drawMatchDisplacement(ft_dist, kp1, None, kp2, None, good)
        img3 = cv2.drawMatches(frame, kp1, prev_frame, kp2, good, None, flags=2)

        # if len(good) > 4:
        #     query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        #     train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        #     matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
        #     matches_mask = mask.ravel().tolist()

            # h, w, c = frame.shape
            # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            # print(pts)
            # dst = cv2.perspectiveTransform(pts, matrix)

            # homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # print(matrix)
            # print(dst)
            # cv2.imshow("Homography", homography)
            # cv2.imshow("mask", mask)
            # cv2.imshow("matches", img3)
        cv2.imshow("frame", frame)
        cv2.imshow("Feature Displacement", ft_dist)

    kp2, des2 = kp1, des1
    prev_frame = frame
    key = cv2.waitKey(1)
    t1 = time.time()
    print(f"FPS: {(t1 - t0)**-1}")

    # if key == ord("s"):
    #     r = cv2.selectROI("select the area", frame)
    #     if r is not roi:
    #         roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    #         rgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #         # edges = cv2.Canny(rgray, 20, 50)
    #         cv2.imshow("roi", rgray)
    #
    #         kp2, des2 = sift.detectAndCompute(rgray, None)

    if key == ord("q"):
        break
cv2.destroyAllWindows()
