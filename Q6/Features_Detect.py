import numpy as np
import cv2

from matplotlib import pyplot as plt
from pathlib import Path
import argparse

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
# Exemples d'usage :
# python3 Q6/Features_Detect.py orb
# python3 Q6/Features_Detect.py kaze --no-show
# python3 Q6/Features_Detect.py orb --nfeatures 400 --scale-factor 1.2 --nlevels 8 --fast-threshold 20 --no-show
# python3 Q6/Features_Detect.py kaze --kaze-threshold 0.002 --kaze-octaves 4 --kaze-octave-layers 4 --upright --no-show

parser = argparse.ArgumentParser(
  description="Détection de points d'intérêt ORB/KAZE sur une paire d'images."
)
parser.add_argument("detector", choices=["orb", "kaze"], help="Détecteur à utiliser")
parser.add_argument("--image1", default="../Image_Pairs/torb_small1.png", help="Chemin vers l'image 1")
parser.add_argument("--image2", default="../Image_Pairs/torb_small2.png", help="Chemin vers l'image 2")
parser.add_argument("--no-show", action="store_true", help="N'ouvre pas la fenêtre matplotlib")

# Paramètres ORB
parser.add_argument("--nfeatures", type=int, default=250, help="Nombre maximal de points ORB")
parser.add_argument("--scale-factor", type=float, default=2.0, help="Facteur d'échelle ORB (>1)")
parser.add_argument("--nlevels", type=int, default=3, help="Nombre de niveaux ORB")
parser.add_argument("--fast-threshold", type=int, default=20, help="Seuil FAST pour ORB")

# Paramètres KAZE
parser.add_argument("--kaze-threshold", type=float, default=0.001, help="Seuil KAZE (>0)")
parser.add_argument("--kaze-octaves", type=int, default=4, help="Nombre d'octaves KAZE")
parser.add_argument("--kaze-octave-layers", type=int, default=4, help="Nombre de couches par octave KAZE")
parser.add_argument("--upright", action="store_true", help="Active KAZE upright (sans orientation)")
args = parser.parse_args()

if args.nfeatures <= 0:
  parser.error("--nfeatures doit être > 0")
if args.nlevels <= 0:
  parser.error("--nlevels doit être > 0")
if args.scale_factor <= 1.0:
  parser.error("--scale-factor doit être > 1")
if args.fast_threshold < 0:
  parser.error("--fast-threshold doit être >= 0")
if args.kaze_threshold <= 0:
  parser.error("--kaze-threshold doit être > 0")
if args.kaze_octaves <= 0:
  parser.error("--kaze-octaves doit être > 0")
if args.kaze_octave_layers <= 0:
  parser.error("--kaze-octave-layers doit être > 0")

def resolve_image_path(path_str):
  p = Path(path_str)
  if p.is_absolute():
    return p
  p_cwd = (Path.cwd() / p).resolve()
  if p_cwd.exists():
    return p_cwd
  p_script = (script_dir / p).resolve()
  return p_script

img1_path = resolve_image_path(args.image1)
img2_path = resolve_image_path(args.image2)

if not img1_path.exists():
  parser.error("image1 introuvable : " + str(img1_path))
if not img2_path.exists():
  parser.error("image2 introuvable : " + str(img2_path))

img1 = cv2.imread(str(img1_path))
if img1 is None:
  parser.error("Impossible de lire image1 : " + str(img1_path))
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread(str(img2_path))
if img2 is None:
  parser.error("Impossible de lire image2 : " + str(img2_path))
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if args.detector == "orb":
  kp1 = cv2.ORB_create(nfeatures = args.nfeatures,#Par défaut : 500
                       scaleFactor = args.scale_factor,#Par défaut : 1.2
                       nlevels = args.nlevels,#Par défaut : 8
                       fastThreshold = args.fast_threshold)
  kp2 = cv2.ORB_create(nfeatures = args.nfeatures,
                       scaleFactor = args.scale_factor,
                       nlevels = args.nlevels,
                       fastThreshold = args.fast_threshold)
  print("Détecteur : ORB")
  print("nfeatures =", args.nfeatures)
  print("scaleFactor =", args.scale_factor)
  print("nlevels =", args.nlevels)
  print("fastThreshold =", args.fast_threshold)
else:
  kp1 = cv2.KAZE_create(upright = args.upright,#Par défaut : false
    		        threshold = args.kaze_threshold,#Par défaut : 0.001
  		        nOctaves = args.kaze_octaves,#Par défaut : 4
		        nOctaveLayers = args.kaze_octave_layers,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = args.upright,#Par défaut : false
	  	        threshold = args.kaze_threshold,#Par défaut : 0.001
		        nOctaves = args.kaze_octaves,#Par défaut : 4
		        nOctaveLayers = args.kaze_octave_layers,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
  print("threshold =", args.kaze_threshold)
  print("nOctaves =", args.kaze_octaves)
  print("nOctaveLayers =", args.kaze_octave_layers)
  print("upright =", args.upright)
  print("diffusivity =", 2)
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection des keypoints
pts1 = kp1.detect(gray1,None)
pts2 = kp2.detect(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection des points d'intérêt :",time,"s")
print("Nombre de points détectés image 1 :", len(pts1))
print("Nombre de points détectés image 2 :", len(pts2))

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

out_dir = repo_root / "docs" / "imgs" / "Q6"
out_dir.mkdir(parents=True, exist_ok=True)
tag = args.detector.lower()
out_img1 = out_dir / f"q6_{tag}_img1.png"
out_img2 = out_dir / f"q6_{tag}_img2.png"
cv2.imwrite(str(out_img1), img1)
cv2.imwrite(str(out_img2), img2)
print("Images sauvegardées :")
print("-", out_img1)
print("-", out_img2)

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

pair_path = out_dir / f"q6_{tag}_pair.png"
plt.savefig(str(pair_path), dpi=150, bbox_inches="tight")
print("-", pair_path)

if not args.no_show:
  plt.show()
