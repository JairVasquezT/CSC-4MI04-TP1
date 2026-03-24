import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import argparse

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent

parser = argparse.ArgumentParser(description="Appariement ORB/KAZE")
parser.add_argument("detector", choices=["orb", "kaze"])
parser.add_argument("strategy", choices=["crosscheck", "ratiotest", "flann"])
parser.add_argument("--image1", default="../Image_Pairs/torb_small1.png")
parser.add_argument("--image2", default="../Image_Pairs/torb_small2.png")
parser.add_argument("--ratio", type=float, default=0.7)
parser.add_argument("--no-show", action="store_true")
parser.add_argument("--nfeatures", type=int, default=500)
parser.add_argument("--scale", type=float, default=1.2)
parser.add_argument("--nlevels", type=int, default=8)
parser.add_argument("--kaze-threshold", type=float, default=0.001)
parser.add_argument("--fast-threshold", type=int, default=20, help="Seuil FAST pour ORB")
parser.add_argument("--kaze-octaves", type=int, default=4, help="Octaves KAZE")
parser.add_argument("--kaze-layers", type=int, default=4, help="Couches KAZE")
parser.add_argument("--upright", action="store_true", help="Désactive l'orientation (plus stable si pas de rotation)")

args = parser.parse_args()

def resolve_path(path_str):
    p = Path(path_str)
    if p.is_absolute(): return p
    p_cwd = (Path.cwd() / p).resolve()
    if p_cwd.exists(): return p_cwd
    return (script_dir / p).resolve()

img1 = cv2.imread(str(resolve_path(args.image1)))
img2 = cv2.imread(str(resolve_path(args.image2)))
if img1 is None or img2 is None:
    print("Error: No se pudieron cargar las imágenes.")
    exit(1)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- Configuración del algoritmo ---
if args.detector == "orb":
    algo = cv2.ORB_create(
        nfeatures=args.nfeatures, 
        scaleFactor=args.scale, 
        nlevels=args.nlevels,
        fastThreshold=args.fast_threshold  # <--- Añadido
    )
    norm_type = cv2.NORM_HAMMING 
else:
    algo = cv2.KAZE_create(
        threshold=args.kaze_threshold,
        nOctaves=args.kaze_octaves,       # <--- Añadido
        nOctaveLayers=args.kaze_layers,   # <--- Añadido
        upright=args.upright              # <--- Añadido
    )
    norm_type = cv2.NORM_L2     

# --- Detección ---
kp1, desc1 = algo.detectAndCompute(gray1, None)
kp2, desc2 = algo.detectAndCompute(gray2, None)

# --- Matching ---
t1 = cv2.getTickCount()
good_matches = []

# Caso 1: CrossCheck (usa .match)
if args.strategy == "crosscheck":
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(desc1, desc2)
    good_matches = sorted(matches, key=lambda x: x.distance)

# Caso 2: FLANN (requiere parámetros distintos para ORB y KAZE)
elif args.strategy == "flann":
    if args.detector == "orb":
        # FLANN para descriptores BINARIOS (ORB)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    else:
        # FLANN para descriptores FLOTANTES (KAZE)
        index_params = dict(algorithm=1, trees=5) # 1 es KDTree
    
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # FLANN necesita que los descriptores de ORB sean uint8 (ya lo son)
    matches = flann.knnMatch(desc1, desc2, k=2)
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < args.ratio * n.distance:
                good_matches.append(m)

# Caso 3: Ratio Test (BFMatcher estándar)
elif args.strategy == "ratiotest":
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < args.ratio * n.distance:
                good_matches.append(m)

t2 = cv2.getTickCount()
matching_time = (t2-t1)/cv2.getTickFrequency()
nb_matches = len(good_matches)

# --- Resultados ---
print(f"RESULT_DATA|{args.detector}|{args.strategy}|{nb_matches}|{matching_time:.4f}")

# Guardado
out_dir = repo_root / "docs" / "imgs" / "Q8"
out_dir.mkdir(parents=True, exist_ok=True)
if args.detector == "orb":
    p_str = f"nf{args.nfeatures}_nl{args.nlevels}_ft{args.fast_threshold}_r{args.ratio}"
else:
    p_str = f"th{args.kaze_threshold}_o{args.kaze_octaves}_l{args.kaze_layers}_u{int(args.upright)}_r{args.ratio}"
out_path = out_dir / f"q8_{args.detector}_{args.strategy}_{p_str}.png"

img_match = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=2)
cv2.imwrite(str(out_path), img_match)

if not args.no_show:
    plt.imshow(img_match)
    plt.title(f"{args.detector.upper()} - {args.strategy} ({nb_matches})")
    plt.show()