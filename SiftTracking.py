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
moving_only = True
background = True

def drawMatchDisplacement(dst, kp1, cl1, kp2, cl2, good, moving_matches=True, mov_thresh=2):
    cl1 = (0, 0, 255) if cl1 is None else cl1
    cl2 = (255, 0, 0) if cl2 is None else cl2
    kp1_idx = []
    kp2_idx = []

    for match in good:
        kp1_id, kp2_id = match.queryIdx, match.trainIdx
        kp1_pos = np.array(kp1[kp1_id].pt).astype(np.int)
        kp2_pos = np.array(kp2[kp2_id].pt).astype(np.int)
        if np.sqrt(np.mean((kp1_pos-kp2_pos)**2)) > mov_thresh:
            dst = cv2.line(dst, kp1_pos, kp2_pos, (0, 255, 0), 2)
            if moving_matches:
                kp1_idx.append(kp1_id)
                kp2_idx.append(kp2_id)
    if moving_matches:
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
        ft_dst = np.zeros(frame.shape, dtype="uint8") if background else frame.copy()
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        ft_dst = drawMatchDisplacement(ft_dst, kp1, (0, 0, 255), kp2, (255, 0, 0), good, moving_only)
        img3 = cv2.drawMatches(frame, kp1, prev_frame, kp2, good, None, flags=2)

        cv2.imshow("frame", frame)
        cv2.imshow("Feature Displacement", ft_dst)

    kp2, des2 = kp1, des1
    prev_frame = frame
    key = cv2.waitKey(1)

    if key == ord("q"):
        cap.release()
        break
    elif key == ord("m"):
        moving_only = not moving_only
    elif key == ord("b"):
        background = not background

    t1 = time.time()
    print(f"FPS: {(t1 - t0) ** -1}")

cv2.destroyAllWindows()
