"""
TP1 Q3 — Gradients par convolution (Sobel).
"""

import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

SHOW = False


def build_paths(image_arg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if os.path.isabs(image_arg):
        image_path = image_arg
    else:
        # Priorite: chemin relatif au script (ex: ../Image_Pairs/FlowerGarden2.png)
        image_path = os.path.normpath(os.path.join(script_dir, image_arg))
        if not os.path.exists(image_path):
            # Fallback pratique pour un argument relatif a la racine du projet.
            image_path = os.path.normpath(os.path.join(project_root, image_arg))

    imgs_dir = os.path.join(project_root, "imgs")
    return image_path, imgs_dir


def robust_limit(values, percentile):
    limit = float(np.percentile(np.abs(values), percentile))
    return limit if limit > 0 else 1.0


def interior_view(values):
    inside = values[1:-1, 1:-1]
    # Robustesse: pour une image trop petite, on retombe sur la vue complete.
    return inside if inside.size > 0 else values


def save_map(data, title, out_path, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_panel(img, ix, iy, grad, mx, my, mg, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im0 = axes[0, 0].imshow(
        img,
        cmap="gray",
        vmin=0.0,
        vmax=255.0,
        interpolation="nearest",
    )
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Ix et Iy doivent rester en float et avec echelle divergente centree en 0.
    im1 = axes[0, 1].imshow(
        ix,
        cmap="seismic",
        vmin=-mx,
        vmax=mx,
        interpolation="nearest",
    )
    axes[0, 1].set_title("Ix = dI/dx")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(
        iy,
        cmap="seismic",
        vmin=-my,
        vmax=my,
        interpolation="nearest",
    )
    axes[1, 0].set_title("Iy = dI/dy")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(
        grad,
        cmap="gray",
        vmin=0.0,
        vmax=mg,
        interpolation="nearest",
    )
    axes[1, 1].set_title("||\u2207I||")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_abs_panel(img, abs_ix, abs_iy, grad, vmax_abs_ix, vmax_abs_iy, mg, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im0 = axes[0, 0].imshow(
        img,
        cmap="gray",
        vmin=0.0,
        vmax=255.0,
        interpolation="nearest",
    )
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        abs_ix,
        cmap="gray",
        vmin=0.0,
        vmax=vmax_abs_ix,
        interpolation="nearest",
    )
    axes[0, 1].set_title("|Ix|")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(
        abs_iy,
        cmap="gray",
        vmin=0.0,
        vmax=vmax_abs_iy,
        interpolation="nearest",
    )
    axes[1, 0].set_title("|Iy|")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(
        grad,
        cmap="gray",
        vmin=0.0,
        vmax=mg,
        interpolation="nearest",
    )
    axes[1, 1].set_title("||\u2207I||")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TP1 Q3: calcul de Ix, Iy et ||grad I|| par Sobel."
    )
    parser.add_argument(
        "--image",
        default=os.path.join("..", "Image_Pairs", "FlowerGarden2.png"),
        help="Chemin image (absolu, relatif a Q3/, ou relatif a la racine).",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile robuste pour l'affichage (defaut: 99).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche les figures Matplotlib en plus de la sauvegarde.",
    )
    args = parser.parse_args()

    if not (0.0 < args.percentile <= 100.0):
        raise ValueError("--percentile doit etre dans ]0, 100].")

    image_path, imgs_dir = build_paths(args.image)
    os.makedirs(imgs_dir, exist_ok=True)

    img_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Image introuvable: {image_path}")
    img = img_u8.astype(np.float64)

    h, w = img.shape
    print(f"Dimension image: {h} lignes x {w} colonnes")

    kx = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ky = np.array(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    ix = cv2.filter2D(img, cv2.CV_64F, kx, borderType=cv2.BORDER_REPLICATE)
    iy = cv2.filter2D(img, cv2.CV_64F, ky, borderType=cv2.BORDER_REPLICATE)
    grad = np.sqrt(ix * ix + iy * iy)

    ix_in = interior_view(ix)
    iy_in = interior_view(iy)
    grad_in = interior_view(grad)

    mx = robust_limit(ix_in, args.percentile)
    my = robust_limit(iy_in, args.percentile)
    mg = float(np.percentile(grad_in, args.percentile))
    if mg <= 0.0:
        mg = 1.0

    n = ix_in.size
    count_ix_neg = int(np.sum(ix_in < 0))
    count_ix_pos = int(np.sum(ix_in > 0))
    count_ix_zero = int(np.sum(ix_in == 0))
    count_iy_neg = int(np.sum(iy_in < 0))
    count_iy_pos = int(np.sum(iy_in > 0))
    count_iy_zero = int(np.sum(iy_in == 0))

    abs_ix = np.abs(ix)
    abs_iy = np.abs(iy)
    ax_in = interior_view(abs_ix)
    ay_in = interior_view(abs_iy)
    vmax_abs_ix = float(np.percentile(ax_in, args.percentile))
    if vmax_abs_ix <= 0.0:
        vmax_abs_ix = 1.0
    vmax_abs_iy = float(np.percentile(ay_in, args.percentile))
    if vmax_abs_iy <= 0.0:
        vmax_abs_iy = 1.0

    print(f"Ix min/max: {ix.min():.6f} / {ix.max():.6f}")
    print(f"Iy min/max: {iy.min():.6f} / {iy.max():.6f}")
    print(f"G  min/max: {grad.min():.6f} / {grad.max():.6f}")
    print(
        f"Percentiles p={args.percentile:.2f}: Mx={mx:.6f}, My={my:.6f}, Mg={mg:.6f}"
    )
    print(
        "Ix interieur: "
        f"neg={count_ix_neg} ({100.0 * count_ix_neg / n:.2f}%), "
        f"pos={count_ix_pos} ({100.0 * count_ix_pos / n:.2f}%), "
        f"zero={count_ix_zero} ({100.0 * count_ix_zero / n:.2f}%)"
    )
    print(
        "Iy interieur: "
        f"neg={count_iy_neg} ({100.0 * count_iy_neg / n:.2f}%), "
        f"pos={count_iy_pos} ({100.0 * count_iy_pos / n:.2f}%), "
        f"zero={count_iy_zero} ({100.0 * count_iy_zero / n:.2f}%)"
    )
    print(
        "Justification affichage: Ix/Iy contiennent des valeurs negatives et positives, "
        "donc affichage float avec echelle divergente centree en 0."
    )
    print(
        f"Percentiles abs p={args.percentile:.2f}: "
        f"vmax(|Ix|)={vmax_abs_ix:.6f}, vmax(|Iy|)={vmax_abs_iy:.6f}"
    )

    out_ix = os.path.join(imgs_dir, "q3_Ix.png")
    out_iy = os.path.join(imgs_dir, "q3_Iy.png")
    out_grad = os.path.join(imgs_dir, "q3_grad_norm.png")
    out_panel = os.path.join(imgs_dir, "q3_grad_panels.png")
    out_abs_ix = os.path.join(imgs_dir, "q3_abs_Ix.png")
    out_abs_iy = os.path.join(imgs_dir, "q3_abs_Iy.png")
    out_abs_panel = os.path.join(imgs_dir, "q3_abs_panels.png")

    save_map(ix, "Ix = dI/dx (Sobel)", out_ix, "seismic", -mx, mx)
    save_map(iy, "Iy = dI/dy (Sobel)", out_iy, "seismic", -my, my)
    save_map(grad, "||\u2207I||", out_grad, "gray", 0.0, mg)
    save_panel(img, ix, iy, grad, mx, my, mg, out_panel)
    save_map(abs_ix, "|Ix|", out_abs_ix, "gray", 0.0, vmax_abs_ix)
    save_map(abs_iy, "|Iy|", out_abs_iy, "gray", 0.0, vmax_abs_iy)
    save_abs_panel(img, abs_ix, abs_iy, grad, vmax_abs_ix, vmax_abs_iy, mg, out_abs_panel)

    print(f"Images sauvegardees dans: {imgs_dir}")
    print(
        "Fichiers: "
        "q3_Ix.png, q3_Iy.png, q3_grad_norm.png, q3_grad_panels.png, "
        "q3_abs_Ix.png, q3_abs_Iy.png, q3_abs_panels.png"
    )

    if SHOW or args.show:
        plt.figure(figsize=(11, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
        plt.title("Image originale")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(ix, cmap="seismic", vmin=-mx, vmax=mx, interpolation="nearest")
        plt.title("Ix = dI/dx")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 3)
        plt.imshow(iy, cmap="seismic", vmin=-my, vmax=my, interpolation="nearest")
        plt.title("Iy = dI/dy")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 4)
        plt.imshow(grad, cmap="gray", vmin=0.0, vmax=mg, interpolation="nearest")
        plt.title("||\u2207I||")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
