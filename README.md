# Fundamental Matrix — Estimation and Epipolar Line Visualization

A general-purpose tool that estimates the **fundamental matrix F** from **two images**
of the same scene taken from different viewpoints, and visualizes the epipolar geometry.
It finds correspondences with SIFT, estimates F with RANSAC + the normalized eight-point
algorithm, and draws the epipolar lines on both images to confirm the correspondences.

What this tool does:

1. Take **two images** of the same scene from different viewpoints.
2. Estimate the **fundamental matrix F** (normalized eight-point algorithm + RANSAC).
3. Draw the **epipolar lines**.
4. **Confirm, quantitatively and visually,** that corresponding points lie on their
   corresponding epipolar lines.

## Setup

Dependencies and Python (3.12) are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a `.venv` and installs the dependencies pinned in `uv.lock`.

## Usage

```bash
# Run on the bundled sample (Aloe stereo pair)
uv run python fundamental_matrix.py

# Run on your own two photos (put them in data/user/ and pass the paths)
uv run python fundamental_matrix.py --img0 data/user/left.jpg --img1 data/user/right.jpg
```

Main options:

| Option | Default | Description |
|---|---|---|
| `--img0`, `--img1` | `data/aloeL.jpg`, `data/aloeR.jpg` | the two input images |
| `--out` | `outputs` | output directory |
| `--ratio` | `0.75` | Lowe ratio test threshold (smaller is stricter) |
| `--max-side` | `1200` | max length of the longer image side [px] (large photos are downscaled automatically) |
| `--num-epilines` | `12` | number of epipolar lines to draw |

### Tips for taking your own photos
- Shoot the same scene twice, moving sideways a little (a translational baseline).
- A textured scene (patterns, relief) yields more feature matches and more stability.
- At least 8 correspondences are required. If there are too few, relax with `--ratio 0.8`.
- Put your own photos in `data/user/` (its contents are git-ignored; the bundled
  `data/aloe*.jpg` samples are tracked).

## Method

The pipeline in `fundamental_matrix.py`:

1. **Features and correspondences** — extract SIFT keypoints/descriptors and obtain
   correspondences with a FLANN-based KNN search (k=2) + Lowe's ratio test.
2. **Estimating F** — reject outliers (RANSAC) with `cv2.findFundamentalMat` (MAGSAC),
   then estimate F on the inliers with a **from-scratch normalized eight-point algorithm**
   (`eight_point`).
   - Hartley normalization (centroid to origin, mean distance √2) → least-squares
     solution via SVD → rank-2 enforcement (det F = 0) → undo the normalization.
   - The linear equations of the eight-point algorithm are solved directly as an
     over-determined system.
3. **Drawing epipolar lines** — with `cv2.computeCorrespondEpilines`, draw, in one image,
   the epipolar line induced by a point in the other image. Points and lines share a color.
4. **Verification** — report the **mean symmetric epipolar distance [px]** on the inliers,
   and compare our F with OpenCV's (after matching scale and sign).

> Convention: this code uses an F that satisfies `pts1ᵀ F pts0 = 0` (same as OpenCV).
> The alternative notation `x0ᵀ F' x1 = 0` refers to the same matrix with `F' = Fᵀ`
> (it only differs in which image is taken as "left").

## Outputs

The following are generated in `outputs/`:

- `matches.png` — RANSAC inlier feature matches.
- `epilines.png` — points and epipolar lines color-coded on both images. You can visually
  confirm that each colored point lies on the **same-colored line** in the other image.
- `results.txt` — a summary of the F matrix, match count, inlier count, and mean
  epipolar distance.

## Results (bundled Aloe pair)

| Metric | Value |
|---|---|
| SIFT correspondences (after ratio test) | 7490 |
| RANSAC inliers | 6575 |
| Mean symmetric epipolar distance (our 8-point) | **0.119 px** |
| Mean symmetric epipolar distance (OpenCV) | 0.119 px |
| Difference between our F and OpenCV's F (normalized Frobenius) | 0.031 |

Correspondences lie on the epipolar lines with sub-pixel accuracy (about 0.12 px), and our
from-scratch eight-point result closely matches OpenCV's.

![epipolar lines](assets/epilines.png)
