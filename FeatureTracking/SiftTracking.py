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
position = np.array([0.0, 0.0])


def drawMatchDisplacement(dst, kp1, cl1, kp2, cl2, good, moving_matches=True, mov_thresh=2, pos_log=None):
    cl1 = (0, 0, 255) if cl1 is None else cl1
    cl2 = (255, 0, 0) if cl2 is None else cl2
    kp1_idx = []
    kp2_idx = []
    displacements = []
    update_displacement = False
    for match in good:
        kp1_id, kp2_id = match.queryIdx, match.trainIdx
        kp1_pos = np.array(kp1[kp1_id].pt).astype(int)
        kp2_pos = np.array(kp2[kp2_id].pt).astype(int)
        disp_vec = kp1_pos-kp2_pos
        eul_dist = np.sqrt(np.mean(disp_vec ** 2))
        if eul_dist > mov_thresh:
            update_displacement = True
            dst = cv2.line(dst, kp1_pos, kp2_pos, (0, 255, 0), 2)
            if moving_matches:
                kp1_idx.append(kp1_id)
                kp2_idx.append(kp2_id)
            if len(pos_log) == 2:
                displacements.append(disp_vec)
    if update_displacement and len(pos_log) == 2:
        displacements = np.array(displacements)
        pos_log += np.mean(displacements, axis=0)

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
    h, w, c = frame.shape
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

        ft_dst = drawMatchDisplacement(ft_dst, kp1, (0, 0, 255), kp2, (255, 0, 0), good, moving_only, 2, position)
        center = position+np.array([w/2, h/2])
        cv2.circle(ft_dst, center.astype(int), 2, (255, 0, 0), -1)
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
    elif key == ord("p"):
        position = np.array([0.0, 0.0])

    t1 = time.time()
    print(f"FPS: {(t1 - t0) ** -1}")

cv2.destroyAllWindows()
