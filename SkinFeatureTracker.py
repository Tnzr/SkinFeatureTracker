import time
import cv2
import numpy as np
from Preprocessing.UsedPipeline.TextureEnhancer import TextureEnhancer
from SemanticSegmentation.segmentation import SemanticSegmentation

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


def limit(n, low, high):
    if n < low:
        return low
    elif n > high:
        return high
    return n


def feature_matcher(des1, des2, match_thresh):
    matches = bf.match(des1, des2)
    good = []
    for match in matches:
        if match.distance < limit(match_thresh, 0, 1) * 50:
            good.append(match)
    return good


if __name__ == '__main__':
    print("Initializing...")
    cap = cv2.VideoCapture(0)
    skin_extractor = SemanticSegmentation(model="SemanticSegmentation/pretrained/model_segmentation_skin_30.pth")
    texture_enhancer = TextureEnhancer()
    roi = []
    orb = cv2.ORB_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)
    cv2.ORB.setMaxFeatures(orb, 1000)
    kp1, des1 = [], None
    kp2, des2 = [], None
    prev_frame = None
    moving_only = True
    background = True
    preprocessing = True
    skin_extraction = True
    position = np.array([0.0, 0.0])
    match_thresh = 0.3
    sharpen = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    fps = 0
    n_features = 0
    print("Done")
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
            segmented_skin = skin_extractor.run(frame) if skin_extraction else frame
            gray = cv2.cvtColor(segmented_skin, cv2.COLOR_BGR2GRAY)
            if preprocessing:
                # equalized_hist = cv2.equalizeHist(gray)
                # gray = cv2.filter2D(src=equalized_hist, ddepth=-1, kernel=sharpen)
                gray = texture_enhancer.run(gray)
            kp1, des1 = orb.detectAndCompute(gray, None)
            n_features = len(kp1)
            if len(kp2) != 0:
                ft_dst = np.zeros(frame.shape, dtype="uint8") if background else gray.copy()
                good = feature_matcher(des1, des2, match_thresh=match_thresh)

                ft_dst = drawMatchDisplacement(ft_dst, kp1, (0, 0, 255), kp2, (255, 0, 0), good, moving_only, 2, position)
                center = position+np.array([w/2, h/2])
                cv2.circle(ft_dst, center.astype(int), 2, (255, 0, 0), -1)
                img3 = cv2.drawMatches(frame, kp1, prev_frame, kp2, good, None, flags=2)

                cv2.imshow("frame", frame)
                ft_dst = cv2.putText(ft_dst, f"FPS: {fps:.2f} - PPP: {preprocessing} - All Features: {not moving_only} - N fts: {n_features}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                ft_dst = cv2.putText(ft_dst, f"S: {int(skin_extraction)}  PPP: {int(preprocessing)}  B: {int(background)}  M: {int(moving_only)}  D", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
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
                preprocessing = not preprocessing
            elif key == ord("s"):
                skin_extraction = not skin_extraction
            elif key == ord("d"):
                position = np.array([0.0, 0.0])
            elif key == ord(","):
                match_thresh -= 0.05
            elif key == ord("."):
                match_thresh += 0.05

            cv2.imshow("frame", frame)
            t1 = time.time()
            fps = (t1 - t0) ** -1

            print(f"FPS: {fps:.2f} Match Thresh: {match_thresh}")

    cv2.destroyAllWindows()
