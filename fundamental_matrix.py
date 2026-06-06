#!/usr/bin/env python3
"""Fundamental matrix estimation and epipolar line visualization.

From two images of the same scene taken from different viewpoints, estimate the
fundamental matrix F and visualize the epipolar geometry. A general-purpose tool:
  1. Find point correspondences with SIFT + ratio test.
  2. Estimate the fundamental matrix F (RANSAC + normalized 8-point algorithm).
  3. Draw the epipolar lines.
  4. Confirm, quantitatively and visually, that corresponding points lie on
     their corresponding epipolar lines.

Usage:
  uv run python fundamental_matrix.py                           # run on the bundled sample
  uv run python fundamental_matrix.py --img0 a.jpg --img1 b.jpg # run on any two images

Convention:
  This code uses an F that satisfies pts1^T F pts0 = 0 (same as OpenCV).
  The alternative notation x0^T F' x1 = 0 refers to the same matrix with
  F' = F^T; it only differs in which image is placed on the "left".
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed (save figures in headless environments)
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# 1. Image loading
# --------------------------------------------------------------------------- #
def load_image(path: str, max_side: int = 1200) -> np.ndarray:
    """Load an image; downscale it if the longer side exceeds max_side (for phone photos)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# 2. Feature detection and matching (SIFT + Lowe ratio test)
# --------------------------------------------------------------------------- #
def detect_and_match(img0: np.ndarray, img1: np.ndarray, ratio: float = 0.75):
    """Detect SIFT features and return correspondences via KNN match + ratio test."""
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    if des0 is None or des1 is None:
        raise RuntimeError("No features were detected.")

    # FLANN (KD-Tree) nearest-neighbour search
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des0, des1, k=2)

    good, pts0, pts1 = [], [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:  # Lowe's ratio test
            good.append(m)
            pts0.append(kp0[m.queryIdx].pt)
            pts1.append(kp1[m.trainIdx].pt)

    return kp0, kp1, good, np.asarray(pts0, np.float64), np.asarray(pts1, np.float64)


# --------------------------------------------------------------------------- #
# 3. Fundamental matrix estimation: normalized 8-point algorithm (from scratch)
# --------------------------------------------------------------------------- #
def normalize_points(pts: np.ndarray):
    """Hartley normalization: move the centroid to the origin and scale so that
    the mean distance from the origin is sqrt(2).

    Returns (normalized homogeneous coords Nx3, transform matrix T 3x3).
    Essential for numerical stability.
    """
    centroid = pts.mean(axis=0)
    dist = np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / dist if dist > 0 else 1.0
    T = np.array([[s, 0, -s * centroid[0]],
                  [0, s, -s * centroid[1]],
                  [0, 0, 1.0]])
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ homog.T).T, T


def eight_point(pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """Estimate F with the normalized 8-point algorithm (pts1^T F pts0 = 0) and
    enforce rank 2.

    The linear equation of the 8-point algorithm
        x0(x1 f00 + y1 f01 + f02) + y0(x1 f10 + y1 f11 + f12)
            + (x1 f20 + y1 f21 + f22) = 0
    is assembled for all correspondences into an over-determined system and
    solved in a least-squares sense via SVD.
    """
    pn0, T0 = normalize_points(pts0)
    pn1, T1 = normalize_points(pts1)

    # Build one row of A per correspondence from pts1^T F pts0 = 0.
    xa, ya = pn1[:, 0], pn1[:, 1]   # left-multiplying point (image1)
    xb, yb = pn0[:, 0], pn0[:, 1]   # right-multiplying point (image0)
    A = np.stack([xa * xb, xa * yb, xa,
                  ya * xb, ya * yb, ya,
                  xb,      yb,      np.ones_like(xa)], axis=1)

    # Least-squares solution of A f = 0: right singular vector of the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 (a fundamental matrix satisfies det F = 0)
    U, S, Vt2 = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vt2

    # Undo the normalization: F = T1^T F_norm T0
    F = T1.T @ F @ T0
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F / np.linalg.norm(F)


# --------------------------------------------------------------------------- #
# 4. Evaluation (symmetric epipolar distance) and comparison utilities
# --------------------------------------------------------------------------- #
def symmetric_epipolar_distance(F: np.ndarray, pts0: np.ndarray,
                                pts1: np.ndarray) -> np.ndarray:
    """Return the symmetric epipolar distance [px] for each correspondence.

    The average of the distance from p1 to the epipolar line l1 = F p0 induced by
    p0, and the distance from p0 to the epipolar line l0 = F^T p1 induced by p1.
    """
    p0 = np.hstack([pts0, np.ones((len(pts0), 1))])
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    l1 = (F @ p0.T).T       # lines in image1
    l0 = (F.T @ p1.T).T     # lines in image0
    num = np.sum(p1 * l1, axis=1)  # p1^T F p0 (= p0^T F^T p1, shared)
    d1 = np.abs(num) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
    d0 = np.abs(num) / np.sqrt(l0[:, 0] ** 2 + l0[:, 1] ** 2)
    return 0.5 * (d0 + d1)


def normalize_for_compare(F: np.ndarray) -> np.ndarray:
    """Set the Frobenius norm to 1 and the sign of the largest-magnitude element
    to positive (for scale/sign-invariant comparison)."""
    F = F / np.linalg.norm(F)
    if F.flatten()[np.argmax(np.abs(F))] < 0:
        F = -F
    return F


# --------------------------------------------------------------------------- #
# 5. Visualization
# --------------------------------------------------------------------------- #
def _line_endpoints(line, w, h):
    """Return the two intersection points of the line ax+by+c=0 with the image
    border (stable for both horizontal and vertical lines)."""
    a, b, c = line
    if abs(b) >= abs(a):  # roughly horizontal line -> parameterize by x
        p0 = (0, -c / b)
        p1 = (w, -(c + a * w) / b)
    else:                 # roughly vertical line -> parameterize by y
        p0 = (-c / a, 0)
        p1 = (-(c + b * h) / a, h)
    return (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))


def draw_epilines(img0, img1, pts0, pts1, F, n=12, seed=0):
    """Select N correspondences and return the two images with color-matched
    epipolar lines drawn in the other image."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts0), size=min(n, len(pts0)), replace=False)
    p0 = pts0[idx].astype(np.float32)
    p1 = pts1[idx].astype(np.float32)
    colors = (plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))[:, :3] * 255).astype(int)

    im0, im1 = img0.copy(), img1.copy()
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]

    # epipolar lines in image1 (from points p0 in image0): whichImage=1
    lines1 = cv2.computeCorrespondEpilines(p0.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    # epipolar lines in image0 (from points p1 in image1): whichImage=2
    lines0 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    for l0, l1, q0, q1, col in zip(lines0, lines1, p0, p1, colors):
        color = (int(col[2]), int(col[1]), int(col[0]))  # RGB -> BGR
        a, b = _line_endpoints(l0, w0, h0)
        cv2.line(im0, a, b, color, 2, cv2.LINE_AA)
        a, b = _line_endpoints(l1, w1, h1)
        cv2.line(im1, a, b, color, 2, cv2.LINE_AA)
        for im, q in ((im0, q0), (im1, q1)):
            cv2.circle(im, (int(q[0]), int(q[1])), 9, (255, 255, 255), -1)
            cv2.circle(im, (int(q[0]), int(q[1])), 7, color, -1)
            cv2.circle(im, (int(q[0]), int(q[1])), 9, (0, 0, 0), 1, cv2.LINE_AA)
    return im0, im1


def save_side_by_side(imgL, imgR, titleL, titleR, suptitle, path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.2))
    for a, img, t in ((ax[0], imgL, titleL), (ax[1], imgR, titleR)):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(t, fontsize=13)
        a.axis("off")
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Fundamental matrix estimation and epipolar line visualization")
    ap.add_argument("--img0", default="data/aloeL.jpg", help="image of viewpoint 0")
    ap.add_argument("--img1", default="data/aloeR.jpg", help="image of viewpoint 1")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    ap.add_argument("--max-side", type=int, default=1200,
                    help="max length of the longer image side [px]")
    ap.add_argument("--num-epilines", type=int, default=12,
                    help="number of epipolar lines to draw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Step 1: two images ----------------------------------------------- #
    img0 = load_image(args.img0, args.max_side)
    img1 = load_image(args.img1, args.max_side)
    print(f"[1] images: {args.img0} {img0.shape[1]}x{img0.shape[0]} / "
          f"{args.img1} {img1.shape[1]}x{img1.shape[0]}")

    # --- Correspondences -------------------------------------------------- #
    kp0, kp1, good, pts0, pts1 = detect_and_match(img0, img1, args.ratio)
    print(f"    SIFT correspondences (after ratio test): {len(pts0)}")
    if len(pts0) < 8:
        raise SystemExit("Fewer than 8 correspondences. Increase --ratio or try other images.")

    # --- Step 2: estimate F (RANSAC for outlier rejection -> 8-point on inliers) --- #
    F_cv, mask = cv2.findFundamentalMat(
        pts0, pts1, cv2.USAC_MAGSAC,
        ransacReprojThreshold=1.0, confidence=0.999, maxIters=10000)
    mask = mask.ravel().astype(bool)
    in0, in1 = pts0[mask], pts1[mask]
    print(f"[2] RANSAC inliers: {int(mask.sum())} / {len(mask)}")

    F_8 = eight_point(in0, in1)  # our own normalized 8-point algorithm (on inliers)

    # Compare against OpenCV's result
    diff = np.linalg.norm(normalize_for_compare(F_cv) - normalize_for_compare(F_8))

    # --- Quantitative check (symmetric epipolar distance) ----------------- #
    d8 = symmetric_epipolar_distance(F_8, in0, in1)
    dcv = symmetric_epipolar_distance(F_cv, in0, in1)
    print(f"    mean symmetric epipolar distance [px]: ours={d8.mean():.3f} "
          f"(median {np.median(d8):.3f}) / OpenCV={dcv.mean():.3f}")
    print(f"    difference between ours and OpenCV F (normalized Frobenius): {diff:.4f}")

    np.set_printoptions(precision=5, suppress=True)
    print("\n[F] ours, normalized 8-point (scaled to F[2,2]=1):")
    print(F_8)
    print("\n[F] OpenCV findFundamentalMat (MAGSAC):")
    print(F_cv / F_cv[2, 2])

    # --- Steps 3 & 4: draw epipolar lines and figures for visual check ---- #
    # match figure (inliers only, up to 80)
    good_in = [g for g, m in zip(good, mask) if m]
    vis_match = cv2.drawMatches(img0, kp0, img1, kp1, good_in[:80], None,
                                matchColor=(0, 255, 0), singlePointColor=None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    p_match = os.path.join(args.out, "matches.png")
    cv2.imwrite(p_match, vis_match)

    # epipolar line figure (using our F)
    e0, e1 = draw_epilines(img0, img1, in0, in1, F_8, n=args.num_epilines)
    p_epi = os.path.join(args.out, "epilines.png")
    save_side_by_side(
        e0, e1,
        "View 0 (image0)",
        "View 1 (image1)",
        f"Epipolar lines: each colored point lies on the same-colored epipolar "
        f"line in the other view (ours, 8-point F, mean epipolar distance {d8.mean():.2f} px)",
        p_epi)

    # Save a text summary as well
    p_txt = os.path.join(args.out, "results.txt")
    with open(p_txt, "w") as f:
        f.write("Fundamental matrix estimation summary\n")
        f.write(f"img0={args.img0}  img1={args.img1}\n")
        f.write(f"SIFT correspondences={len(pts0)}  RANSAC inliers={int(mask.sum())}\n")
        f.write(f"mean symmetric epipolar distance [px]: ours={d8.mean():.4f}  "
                f"OpenCV={dcv.mean():.4f}\n")
        f.write(f"difference between ours and OpenCV F (normalized)={diff:.5f}\n\n")
        f.write("F (ours, normalized 8-point, F[2,2]=1):\n" + np.array2string(F_8) + "\n\n")
        f.write("F (OpenCV MAGSAC, F[2,2]=1):\n" + np.array2string(F_cv / F_cv[2, 2]) + "\n")

    print(f"\n[3,4] outputs: {p_match} / {p_epi} / {p_txt}")
    print("      In epilines.png, check that each colored point lies on the "
          "same-colored epipolar line in the other image.")


if __name__ == "__main__":
    main()
