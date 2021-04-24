import cv2
import numpy as np

def university_keypoints():
    img1 = cv2.imread("Resources/UoH.jpg")

    sift = cv2.SIFT_create()
    kp1, descriptors = sift.detectAndCompute(img1, None)

    count = 1
    for kp in kp1:
        # if count % 2 == 0:
        #     count += 1
        #     continue
        # else:
        angle = np.deg2rad(kp.angle)
        size = round(kp.size / 2)
        if size < 5:
            continue
        x_cord = kp.pt[0]
        y_cord = kp.pt[1]
        x1 = round(kp.pt[0] + (size * np.cos(angle)))
        y1 = round(kp.pt[1] + (size * np.sin(angle)))
        cv2.arrowedLine(img1, (round(x_cord), round(y_cord)), (x1, y1), (0, 0, 255), 1)
        cv2.circle(img1, (round(x_cord), round(y_cord)), size, (255, 0, 0), 1)

        count += 1

    return img1

