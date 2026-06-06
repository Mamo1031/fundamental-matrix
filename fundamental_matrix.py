#!/usr/bin/env python3
"""基礎行列 (Fundamental Matrix) の推定とエピポーラ線の描画.

同一シーンを別視点から撮った 2 枚の画像から基礎行列 F を推定し、
エピポーラ幾何を可視化する汎用ツール:
  1. SIFT + ratio test で対応点を求める
  2. 基礎行列 F を推定する (RANSAC + 正規化 8 点法)
  3. エピポーラ線を描画する
  4. 対応点が対応するエピポーラ線上に乗ることを定量・目視で確認する

使い方:
  uv run python fundamental_matrix.py                           # 付属サンプルで実行
  uv run python fundamental_matrix.py --img0 a.jpg --img1 b.jpg # 任意の 2 枚で実行

規約:
  本コードは pts1^T F pts0 = 0 を満たす F を用いる (OpenCV と同じ規約)。
  x0^T F' x1 = 0 の表記とは F' = F^T で同じ行列を指しており、
  どちらの画像を「左」に置くかの違いにすぎない。
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # ディスプレイ不要 (ヘッドレス環境で図を保存)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _setup_japanese_font() -> None:
    """日本語フォントがあれば matplotlib に登録する (図のタイトルの豆腐化を防ぐ)。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 1. 画像の読み込み
# --------------------------------------------------------------------------- #
def load_image(path: str, max_side: int = 1200) -> np.ndarray:
    """画像を読み込み、長辺が max_side を超える場合は縮小する (自分の写真対応)。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- #
# 2. 特徴点検出とマッチング (SIFT + Lowe ratio test)
# --------------------------------------------------------------------------- #
def detect_and_match(img0: np.ndarray, img1: np.ndarray, ratio: float = 0.75):
    """SIFT で特徴点を検出し、KNN マッチ + ratio test で対応点を返す。"""
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    if des0 is None or des1 is None:
        raise RuntimeError("特徴点が検出できませんでした。")

    # FLANN (KD-Tree) による近傍探索
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
# 3. 基礎行列の推定: 正規化 8 点法 (自前実装)
# --------------------------------------------------------------------------- #
def normalize_points(pts: np.ndarray):
    """Hartley 正規化: 重心を原点へ、原点からの平均距離を sqrt(2) にする。

    返り値: (正規化済み同次座標 Nx3, 変換行列 T 3x3)。数値安定性に必須。
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
    """正規化 8 点法で F を推定 (pts1^T F pts0 = 0)。rank-2 を強制する。

    8 点法の線形方程式
        x0(x1 f00 + y1 f01 + f02) + y0(x1 f10 + y1 f11 + f12)
            + (x1 f20 + y1 f21 + f22) = 0
    を全対応点について並べた over-determined な系を SVD で最小二乗的に解く。
    """
    pn0, T0 = normalize_points(pts0)
    pn1, T1 = normalize_points(pts1)

    # pts1^T F pts0 = 0 から、各対応について A の 1 行を作る。
    xa, ya = pn1[:, 0], pn1[:, 1]   # 左から掛ける点 (image1)
    xb, yb = pn0[:, 0], pn0[:, 1]   # 右から掛ける点 (image0)
    A = np.stack([xa * xb, xa * yb, xa,
                  ya * xb, ya * yb, ya,
                  xb,      yb,      np.ones_like(xa)], axis=1)

    # A f = 0 の最小二乗解 = 最小特異値に対応する右特異ベクトル
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # rank-2 強制 (基礎行列は det F = 0 を満たす)
    U, S, Vt2 = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vt2

    # 正規化を元に戻す: F = T1^T F_norm T0
    F = T1.T @ F @ T0
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F / np.linalg.norm(F)


# --------------------------------------------------------------------------- #
# 4. 評価 (対称エピポーラ距離) と比較ユーティリティ
# --------------------------------------------------------------------------- #
def symmetric_epipolar_distance(F: np.ndarray, pts0: np.ndarray,
                                pts1: np.ndarray) -> np.ndarray:
    """各対応点の対称エピポーラ距離 [px] を返す。

    点 p1 と、p0 が引くエピポーラ線 l1 = F p0 との距離、および
    点 p0 と、p1 が引くエピポーラ線 l0 = F^T p1 との距離の平均。
    """
    p0 = np.hstack([pts0, np.ones((len(pts0), 1))])
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    l1 = (F @ p0.T).T       # image1 上の線
    l0 = (F.T @ p1.T).T     # image0 上の線
    num = np.sum(p1 * l1, axis=1)  # p1^T F p0 (= p0^T F^T p1, 共通)
    d1 = np.abs(num) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
    d0 = np.abs(num) / np.sqrt(l0[:, 0] ** 2 + l0[:, 1] ** 2)
    return 0.5 * (d0 + d1)


def normalize_for_compare(F: np.ndarray) -> np.ndarray:
    """Frobenius ノルムを 1 に、最大絶対値要素の符号を正に揃える (比較用)。"""
    F = F / np.linalg.norm(F)
    if F.flatten()[np.argmax(np.abs(F))] < 0:
        F = -F
    return F


# --------------------------------------------------------------------------- #
# 5. 可視化
# --------------------------------------------------------------------------- #
def _line_endpoints(line, w, h):
    """直線 ax+by+c=0 と画像枠の交点 2 点を返す (水平/垂直どちらでも安定)。"""
    a, b, c = line
    if abs(b) >= abs(a):  # 横向きの線 -> x でパラメータ化
        p0 = (0, -c / b)
        p1 = (w, -(c + a * w) / b)
    else:                 # 縦向きの線 -> y でパラメータ化
        p0 = (-c / a, 0)
        p1 = (-(c + b * h) / a, h)
    return (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))


def draw_epilines(img0, img1, pts0, pts1, F, n=12, seed=0):
    """N 組の対応点を選び、相手画像にエピポーラ線を色対応で描画した 2 枚を返す。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts0), size=min(n, len(pts0)), replace=False)
    p0 = pts0[idx].astype(np.float32)
    p1 = pts1[idx].astype(np.float32)
    colors = (plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))[:, :3] * 255).astype(int)

    im0, im1 = img0.copy(), img1.copy()
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]

    # image1 上のエピポーラ線 (image0 の点 p0 から): whichImage=1
    lines1 = cv2.computeCorrespondEpilines(p0.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    # image0 上のエピポーラ線 (image1 の点 p1 から): whichImage=2
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
# メイン
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="基礎行列の推定とエピポーラ線の描画")
    ap.add_argument("--img0", default="data/aloeL.jpg", help="視点0の画像")
    ap.add_argument("--img1", default="data/aloeR.jpg", help="視点1の画像")
    ap.add_argument("--out", default="outputs", help="出力ディレクトリ")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test の閾値")
    ap.add_argument("--max-side", type=int, default=1200, help="入力画像の長辺上限[px]")
    ap.add_argument("--num-epilines", type=int, default=12, help="描画するエピポーラ線の本数")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _setup_japanese_font()

    # --- ステップ1: 2 枚の画像 -------------------------------------------- #
    img0 = load_image(args.img0, args.max_side)
    img1 = load_image(args.img1, args.max_side)
    print(f"[1] 画像: {args.img0} {img0.shape[1]}x{img0.shape[0]} / "
          f"{args.img1} {img1.shape[1]}x{img1.shape[0]}")

    # --- 対応点 ----------------------------------------------------------- #
    kp0, kp1, good, pts0, pts1 = detect_and_match(img0, img1, args.ratio)
    print(f"    SIFT 対応点 (ratio test 後): {len(pts0)} 組")
    if len(pts0) < 8:
        raise SystemExit("対応点が 8 組未満です。--ratio を上げるか別の画像を試してください。")

    # --- ステップ2: F の推定 (RANSAC で外れ値除去 → inlier で 8 点法) ----- #
    F_cv, mask = cv2.findFundamentalMat(
        pts0, pts1, cv2.USAC_MAGSAC,
        ransacReprojThreshold=1.0, confidence=0.999, maxIters=10000)
    mask = mask.ravel().astype(bool)
    in0, in1 = pts0[mask], pts1[mask]
    print(f"[2] RANSAC inlier: {int(mask.sum())} / {len(mask)} 組")

    F_8 = eight_point(in0, in1)  # 自前の正規化 8 点法 (inlier に対して)

    # OpenCV の結果と比較
    diff = np.linalg.norm(normalize_for_compare(F_cv) - normalize_for_compare(F_8))

    # --- 定量検証 (対称エピポーラ距離) ----------------------------------- #
    d8 = symmetric_epipolar_distance(F_8, in0, in1)
    dcv = symmetric_epipolar_distance(F_cv, in0, in1)
    print(f"    平均対称エピポーラ距離 [px]: 自前8点法={d8.mean():.3f} (中央値{np.median(d8):.3f}) / "
          f"OpenCV={dcv.mean():.3f}")
    print(f"    自前F と OpenCV F の差 (正規化後 Frobenius): {diff:.4f}")

    np.set_printoptions(precision=5, suppress=True)
    print("\n[F] 自前 正規化8点法 (F[2,2]=1 に正規化):")
    print(F_8)
    print("\n[F] OpenCV findFundamentalMat (MAGSAC):")
    print(F_cv / F_cv[2, 2])

    # --- ステップ3 & 4: エピポーラ線の描画と目視確認用の図 ---------------- #
    # マッチ図 (inlier のみ, 最大 80 本)
    good_in = [g for g, m in zip(good, mask) if m]
    vis_match = cv2.drawMatches(img0, kp0, img1, kp1, good_in[:80], None,
                                matchColor=(0, 255, 0), singlePointColor=None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    p_match = os.path.join(args.out, "matches.png")
    cv2.imwrite(p_match, vis_match)

    # エピポーラ線図 (自前 F を使用)
    e0, e1 = draw_epilines(img0, img1, in0, in1, F_8, n=args.num_epilines)
    p_epi = os.path.join(args.out, "epilines.png")
    save_side_by_side(
        e0, e1,
        "視点0 (image0)",
        "視点1 (image1)",
        f"エピポーラ線 — 各色の点は、相手画像の同じ色のエピポーラ線上に乗る"
        f"（自前8点法 F, 平均エピポーラ距離 {d8.mean():.2f}px）",
        p_epi)

    # 結果サマリをテキストでも保存
    p_txt = os.path.join(args.out, "results.txt")
    with open(p_txt, "w") as f:
        f.write("基礎行列推定 結果サマリ\n")
        f.write(f"img0={args.img0}  img1={args.img1}\n")
        f.write(f"SIFT対応点={len(pts0)}  RANSAC inlier={int(mask.sum())}\n")
        f.write(f"平均対称エピポーラ距離[px]: 自前8点法={d8.mean():.4f}  OpenCV={dcv.mean():.4f}\n")
        f.write(f"自前F と OpenCV F の差(正規化後)={diff:.5f}\n\n")
        f.write("F (自前 正規化8点法, F[2,2]=1):\n" + np.array2string(F_8) + "\n\n")
        f.write("F (OpenCV MAGSAC, F[2,2]=1):\n" + np.array2string(F_cv / F_cv[2, 2]) + "\n")

    print(f"\n[3,4] 出力: {p_match} / {p_epi} / {p_txt}")
    print("      epilines.png で、各色の点が相手画像の同色エピポーラ線上に乗ることを確認してください。")


if __name__ == "__main__":
    main()
