import argparse
import csv
import itertools
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Exemples d'execution avec CSV :
# python3 Q5/Harris2.py --window-size 5 --alpha 0.04 --tag simple
# python3 Q5/Harris2.py --window-sizes 3,5,7,9 --alphas 0.04,0.05,0.06 --no-show --tag q5
# python3 Q5/Harris2.py --window-sizes 3,5 --alphas 0.04,0.06 --no-show --csv-path docs/imgs/q5_harris_metrics.csv

DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "Image_Pairs", "Graffiti0.png")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "imgs")

CSV_FIELDS = [
    "image_path",
    "image_name",
    "image_height",
    "image_width",
    "window_size",
    "alpha",
    "maxloc_size",
    "threshold_rel",
    "time_s",
    "cpp",
    "nb_points_detected",
    "nb_pixels_marked",
    "output_original_path",
    "output_theta_path",
    "output_points_path",
    "tag",
]


def _validate_odd_positive(value, name):
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} doit etre un entier impair strictement positif.")


def _parse_csv_ints(value, name):
    items = [item.strip() for item in value.split(",")]
    if not items or any(item == "" for item in items):
        raise ValueError(f"{name} doit etre une liste non vide (ex: 3,5,7).")
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Valeur invalide dans {name}: '{item}'.") from exc
    return parsed


def _parse_csv_floats(value, name):
    items = [item.strip() for item in value.split(",")]
    if not items or any(item == "" for item in items):
        raise ValueError(f"{name} doit etre une liste non vide (ex: 0.04,0.05).")
    parsed = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError as exc:
            raise ValueError(f"Valeur invalide dans {name}: '{item}'.") from exc
    return parsed


def _format_float_for_name(value):
    txt = f"{value:.12g}"
    if "e" in txt or "E" in txt:
        txt = f"{value:.12f}".rstrip("0").rstrip(".")
    return txt.replace("-", "m").replace(".", "p")


def _sanitize_tag(tag):
    if tag is None:
        return None
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in tag.strip())
    return cleaned or None


def _default_csv_path(output_dir, tag):
    clean_tag = _sanitize_tag(tag)
    if clean_tag is None:
        return os.path.join(output_dir, "q5_harris_metrics.csv")
    return os.path.join(output_dir, f"q5_harris_metrics_{clean_tag}.csv")


def save_results_csv(csv_path, rows, append):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if (not append) or (not file_exists):
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_harris(image_path, window_size, alpha, maxloc_size, threshold_rel, output_dir, show, tag):
    #Lecture image en niveau de gris et conversion en float64
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    img = np.float64(img_gray)
    (h, w) = img.shape
    image_height = h
    image_width = w
    print("Dimension de l'image :", h, "lignes x", w, "colonnes")
    print("Type de l'image :", img.dtype)

    #Début du calcul
    t1 = cv2.getTickCount()
    Theta = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

    # Mettre ici le calcul de la fonction d'intérêt de Harris
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    taille_W = window_size
    Sxx = cv2.boxFilter(Ix2, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)
    Syy = cv2.boxFilter(Iy2, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)
    Sxy = cv2.boxFilter(Ixy, ddepth=-1, ksize=(taille_W, taille_W), normalize=False)

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    alpha_harris = alpha
    Theta = det - alpha_harris * (trace * trace)

    # Calcul des maxima locaux et seuillage
    Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    d_maxloc = maxloc_size
    seuil_relatif = threshold_rel
    se = np.ones((d_maxloc, d_maxloc), np.uint8)
    Theta_dil = cv2.dilate(Theta, se)
    #Suppression des non-maxima-locaux
    Theta_maxloc[Theta < Theta_dil] = 0.0
    #On néglige également les valeurs trop faibles
    Theta_maxloc[Theta < seuil_relatif * Theta.max()] = 0.0
    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    cpp = (t2 - t1) / (h * w)
    print("Mon calcul des points de Harris :", time, "s")
    print("Nombre de cycles par pixel :", cpp, "cpp")

    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.title("Image originale")

    plt.subplot(132)
    plt.imshow(Theta, cmap="gray")
    plt.title("Fonction de Harris")

    se_croix = np.uint8([[1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0], [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)
    #Relecture image pour affichage couleur
    Img_pts = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if Img_pts is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    (h, w, c) = Img_pts.shape
    print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")
    print("Type de l'image :", Img_pts.dtype)
    #On affiche les points (croix) en rouge
    Img_pts[Theta_ml_dil > 0] = [255, 0, 0]
    plt.subplot(133)
    plt.imshow(Img_pts)
    plt.title("Points de Harris")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)

    param_suffix = (
        f"ws{window_size}_a{_format_float_for_name(alpha)}"
        f"_m{maxloc_size}_t{_format_float_for_name(threshold_rel)}"
    )
    clean_tag = _sanitize_tag(tag)
    if clean_tag is not None:
        param_suffix = f"{clean_tag}_{param_suffix}"

    orig_path = os.path.join(output_dir, f"q5_harris_orig_{param_suffix}.png")
    theta_path = os.path.join(output_dir, f"q5_harris_theta_{param_suffix}.png")
    points_path = os.path.join(output_dir, f"q5_harris_points_{param_suffix}.png")

    Theta_u8 = cv2.normalize(Theta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(theta_path, Theta_u8)
    cv2.imwrite(points_path, Img_pts)

    if show:
        plt.show()
    plt.close(fig)

    nb_points = np.count_nonzero(Theta_maxloc > 0)
    print("Nombre de points de Harris détectés :", nb_points)

    nb_points_affiches = np.count_nonzero(Theta_ml_dil > 0)
    print("Nombre de pixels marqués pour l'affichage :", nb_points_affiches)

    return {
        "image_path": os.path.abspath(image_path),
        "image_name": os.path.basename(image_path),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "window_size": int(window_size),
        "alpha": float(alpha),
        "maxloc_size": int(maxloc_size),
        "threshold_rel": float(threshold_rel),
        "time_s": float(time),
        "cpp": float(cpp),
        "nb_points_detected": int(nb_points),
        "nb_pixels_marked": int(nb_points_affiches),
        "output_original_path": os.path.abspath(orig_path),
        "output_theta_path": os.path.abspath(theta_path),
        "output_points_path": os.path.abspath(points_path),
        "tag": clean_tag if clean_tag is not None else "",
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Détecteur de Harris (simple, batch et export CSV).")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Chemin de l'image d'entrée.")
    parser.add_argument("--window-size", type=int, default=5, help="Taille impaire de la fenêtre W.")
    parser.add_argument("--alpha", type=float, default=0.04, help="Paramètre alpha de Harris (> 0).")
    parser.add_argument("--maxloc-size", type=int, default=3, help="Taille impaire du voisinage des maxima locaux.")
    parser.add_argument("--threshold-rel", type=float, default=0.01, help="Seuil relatif appliqué à Theta.max().")
    parser.add_argument("--no-show", action="store_true", help="Désactive plt.show() pour les exécutions batch.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Répertoire de sauvegarde des images.")
    parser.add_argument("--tag", default=None, help="Tag optionnel ajouté au nom des fichiers.")
    parser.add_argument("--window-sizes", default=None, help="Liste de tailles de fenêtre, ex: 3,5,7,9.")
    parser.add_argument("--alphas", default=None, help="Liste de alphas, ex: 0.04,0.05,0.06.")
    parser.add_argument("--csv-path", default=None, help="Chemin du fichier CSV de sortie.")
    parser.add_argument("--append-csv", action="store_true", help="Ajoute les lignes au CSV existant.")
    parser.add_argument("--no-csv", action="store_true", help="Désactive la sauvegarde CSV.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        _validate_odd_positive(args.window_size, "--window-size")
        _validate_odd_positive(args.maxloc_size, "--maxloc-size")
        if args.alpha <= 0:
            raise ValueError("--alpha doit etre strictement positif.")
        if args.threshold_rel < 0:
            raise ValueError("--threshold-rel doit etre >= 0.")

        if args.window_sizes is None:
            window_sizes = [args.window_size]
        else:
            window_sizes = _parse_csv_ints(args.window_sizes, "--window-sizes")
            for ws in window_sizes:
                _validate_odd_positive(ws, "--window-sizes")

        if args.alphas is None:
            alphas = [args.alpha]
        else:
            alphas = _parse_csv_floats(args.alphas, "--alphas")
            if any(alpha_val <= 0 for alpha_val in alphas):
                raise ValueError("--alphas doit contenir uniquement des valeurs > 0.")
    except ValueError as exc:
        parser.error(str(exc))

    total = len(window_sizes) * len(alphas)
    image_path = args.image
    output_dir = args.output_dir
    results = []

    for idx, (window_size, alpha) in enumerate(itertools.product(window_sizes, alphas), start=1):
        print(f"\n===== Expérience {idx}/{total} =====")
        res = run_harris(
            image_path=image_path,
            window_size=window_size,
            alpha=alpha,
            maxloc_size=args.maxloc_size,
            threshold_rel=args.threshold_rel,
            output_dir=output_dir,
            show=not args.no_show,
            tag=args.tag,
        )
        results.append(res)
        print(
            "Résumé expérience:"
            f" image={res['image_path']}"
            f" window-size={res['window_size']}"
            f" alpha={res['alpha']}"
            f" maxloc-size={res['maxloc_size']}"
            f" threshold-rel={res['threshold_rel']}"
            f" time={res['time_s']:.9f}s"
            f" cpp={res['cpp']:.9f}"
            f" points={res['nb_points_detected']}"
            f" pixels-marques={res['nb_pixels_marked']}"
            f" orig={res['output_original_path']}"
            f" theta={res['output_theta_path']}"
            f" points-img={res['output_points_path']}"
        )

    if not args.no_csv:
        if args.csv_path is None:
            csv_path = _default_csv_path(output_dir, args.tag)
        else:
            csv_path = args.csv_path
        save_results_csv(csv_path=csv_path, rows=results, append=args.append_csv)
        print("CSV sauvegardé dans :", os.path.abspath(csv_path))
        print("Nombre total d'expériences enregistrées :", len(results))
    else:
        print("Sauvegarde CSV désactivée (--no-csv).")


if __name__ == "__main__":
    main()
