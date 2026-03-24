import cv2
import numpy as np
import argparse


img1 = cv2.imread("../Image_Pairs/torb_small1.png", 0)
rows, cols = img1.shape


M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
M[0, 2] += 20 
img2 = cv2.warpAffine(img1, M, (cols, rows))


orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)


correct_matches = 0
dist_threshold = 3.0 

for m in matches:

    p1 = kp1[m.queryIdx].pt

    p2 = kp2[m.trainIdx].pt
    

    expected_p2 = np.dot(M, (p1[0], p1[1], 1))
    
    
    error = np.sqrt((expected_p2[0] - p2[0])**2 + (expected_p2[1] - p2[1])**2)
    
    if error < dist_threshold:
        correct_matches += 1

precision = (correct_matches / len(matches)) * 100 if len(matches) > 0 else 0
print(f"Total Matches: {len(matches)}")
print(f"Matches Correctos (Inliers): {correct_matches}")
print(f"Precisión: {precision:.2f}%")

out_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

out_path = "q9_validation_orb.png"
cv2.imwrite(out_path, out_img)
print(f"Imagen de validación guardada en: {out_path}")

# Si quieres verla ahora mismo:
import matplotlib.pyplot as plt
plt.imshow(out_img)
plt.title(f"Evaluación Q9: Precisión {precision:.2f}%")
plt.show()