import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
image_path = os.path.join(os.path.dirname(__file__), '..', 'Image_Pairs', 'Graffiti0.png')
img_gray = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"Image introuvable : {image_path}")
img=np.float64(img_gray)
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

# Mettre ici le calcul de la fonction d'intérêt de Harris
Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy

taille_W = 5
Sxx = cv2.boxFilter(Ix2, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)
Syy = cv2.boxFilter(Iy2, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)
Sxy = cv2.boxFilter(Ixy, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)

det = Sxx * Syy - Sxy * Sxy
trace = Sxx + Syy
alpha = 0.04
Theta = det - alpha * (trace * trace)

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread(image_path,cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'imgs')
os.makedirs(output_dir, exist_ok=True)

Theta_u8 = cv2.normalize(Theta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 'q4_harris_image_originale.png'), img_gray)
cv2.imwrite(os.path.join(output_dir, 'q4_harris_fonction_harris.png'), Theta_u8)
cv2.imwrite(os.path.join(output_dir, 'q4_harris_points_harris.png'), Img_pts)

plt.show()

nb_points = np.count_nonzero(Theta_maxloc > 0)
print("Nombre de points de Harris détectés :", nb_points)

nb_points_affiches = np.count_nonzero(Theta_ml_dil > 0)
print("Nombre de pixels marqués pour l'affichage :", nb_points_affiches)
