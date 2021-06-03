import cv2
import numpy as np

def run_warp():
    sourceIMG = cv2.imread("Resources/Dylan.jpg")
    targetIMG = cv2.imread("Resources/frames.jpg")

    shap = np.shape(sourceIMG)
    rows = shap[0]
    cols = shap[1]

    pointSet_Source_Perspective = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    pointSet_Source_Affine = np.float32([[0, 0], [cols, 0], [cols, rows]])

    pointSet_Target_Perspective = np.float32([[195, 56], [494, 159], [431, 498], [39, 183]])
    pointSet_Target_Affine = np.float32([[551, 220], [844, 66], [902, 301]])

    shap = np.shape(targetIMG)
    rows = shap[0]
    cols = shap[1]
    affineTransformation = cv2.getAffineTransform(pointSet_Source_Affine, pointSet_Target_Affine)
    imgOutput = cv2.warpAffine(sourceIMG, affineTransformation, (cols, rows))

    perspectiveTransformation = cv2.getPerspectiveTransform(pointSet_Source_Perspective, pointSet_Target_Perspective)
    imgOutput2 = cv2.warpPerspective(sourceIMG, perspectiveTransformation, (cols, rows))

    newTarget = targetIMG + imgOutput
    newTarget = newTarget + imgOutput2
    return newTarget
