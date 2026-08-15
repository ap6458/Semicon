import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCALES = np.arange(0.090, 0.1021, 0.002)
ROTATIONS = np.arange(-3.0, 3.01, 1.0)

PATCH_SIZE = 128
TOP_TEMPLATE_PEAKS = 250

MIN_PITCH = 15
MAX_PITCH = 120
NEIGHBOUR_RADIUS = 2


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise RuntimeError(f"Could not load image: {path}")
    return img


def normalize(img):
    img = np.asarray(img, dtype=np.float32)
    if img.size == 0:
        raise RuntimeError("Cannot normalize empty image")

    lo = float(np.percentile(img, 1))
    hi = float(np.percentile(img, 99))

    if hi <= lo:
        out = np.zeros_like(img, dtype=np.float32)
    else:
        out = (img - np.float32(lo)) / np.float32(hi - lo)
        out = np.clip(out, 0.0, 1.0)

    return np.ascontiguousarray(out, dtype=np.float32)


def make_template(reference, scale, rotation):
    h, w = reference.shape

    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))

    template = cv2.resize(
        reference, (nw, nh), interpolation=cv2.INTER_AREA
    )

    matrix = cv2.getRotationMatrix2D(
        (nw / 2.0, nh / 2.0), rotation, 1.0
    )

    template = cv2.warpAffine(
        template,
        matrix,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return np.ascontiguousarray(template)


def template_score(search, template):
    if search is None or template is None:
        return None
    if search.size == 0 or template.size == 0:
        return None

    search_n = np.ascontiguousarray(normalize(search), dtype=np.float32)
    template_n = np.ascontiguousarray(normalize(template), dtype=np.float32)

    if (
        template_n.shape[0] > search_n.shape[0]
        or template_n.shape[1] > search_n.shape[1]
    ):
        return None

    return cv2.matchTemplate(
        search_n, template_n, cv2.TM_CCOEFF_NORMED
    )


def extract_peaks(score, k=100, suppression=12):
    work = score.copy()
    result = []

    for _ in range(k):
        _, value, _, location = cv2.minMaxLoc(work)

        if not np.isfinite(value):
            break

        x, y = location
        result.append((float(x), float(y), float(value)))

        x0 = max(0, x - suppression)
        x1 = min(work.shape[1], x + suppression + 1)
        y0 = max(0, y - suppression)
        y1 = min(work.shape[0], y + suppression + 1)

        work[y0:y1, x0:x1] = -np.inf

    return result


def estimate_pitch(image):
    """
    Estimate the fundamental DRAM lattice pitch.

    Autocorrelation can select a smaller harmonic instead of the
    actual DRAM cell pitch. Search the expected fundamental range
    first and reject an implausible X/Y mismatch.
    """

    img = np.asarray(image, dtype=np.float32)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    vertical_profile = np.mean(np.abs(gx), axis=0)
    horizontal_profile = np.mean(np.abs(gy), axis=1)

    def find_fundamental(profile):
        profile = np.asarray(profile, dtype=np.float32)
        profile -= profile.mean()

        std = profile.std()
        if std < 1e-6:
            return None

        profile /= std

        ac = np.correlate(profile, profile, mode="full")
        ac = ac[len(ac) // 2:]

        # Ignore small harmonics. For this dataset the useful
        # fundamental DRAM pitch is around 35-70 pixels.
        lo = 35
        hi = min(70, len(ac) - 1)

        if hi <= lo:
            return None

        region = ac[lo:hi + 1]
        peaks = []

        for i in range(1, len(region) - 1):
            if (
                region[i] >= region[i - 1]
                and region[i] >= region[i + 1]
            ):
                peaks.append(
                    (float(region[i]), i + lo)
                )

        if not peaks:
            return float(lo + np.argmax(region))

        peaks.sort(key=lambda p: p[0], reverse=True)
        return float(peaks[0][1])

    px = find_fundamental(vertical_profile)
    py = find_fundamental(horizontal_profile)

    if px is None:
        px = 47.0
    if py is None:
        py = 47.0

    # The generated DRAM layout is approximately square-periodic.
    # If one axis selects a different harmonic, use a common pitch.
    if abs(px - py) > 12:
        common = 0.5 * (px + py)
        px = common
        py = common

    return float(px), float(py)


def crop(image, cx, cy, size):
    half = size // 2

    x = int(round(cx))
    y = int(round(cy))

    x0 = x - half
    y0 = y - half
    x1 = x0 + size
    y1 = y0 + size

    if (
        x0 < 0 or y0 < 0
        or x1 > image.shape[1]
        or y1 > image.shape[0]
    ):
        return None

    patch = image[y0:y1, x0:x1]

    if patch.shape != (size, size):
        return None

    return patch.astype(np.float32)


def standardize(patch):
    patch = patch.astype(np.float32)

    mean = patch.mean()
    std = patch.std()

    if std < 1e-6:
        return patch - mean

    return (patch - mean) / std


def periodic_anomaly(search, cx, cy, pitch_x, pitch_y):
    center = crop(search, cx, cy, PATCH_SIZE)
    if center is None:
        return None

    center = standardize(center)
    neighbours = []

    for dx in range(-NEIGHBOUR_RADIUS, NEIGHBOUR_RADIUS + 1):
        for dy in range(-NEIGHBOUR_RADIUS, NEIGHBOUR_RADIUS + 1):
            if dx == 0 and dy == 0:
                continue

            nx = cx + dx * pitch_x
            ny = cy + dy * pitch_y

            patch = crop(search, nx, ny, PATCH_SIZE)
            if patch is not None:
                neighbours.append(standardize(patch))

    if len(neighbours) < 4:
        return None

    stack = np.stack(neighbours, axis=0)
    expected = np.median(stack, axis=0)
    residual = np.abs(center - expected)

    mean_residual = float(np.mean(residual))
    high_residual = float(np.percentile(residual, 97))

    p90 = np.percentile(residual, 90)
    strong_fraction = float(np.mean(residual > p90))

    center_u8 = np.uint8(np.clip(center * 35 + 128, 0, 255))
    expected_u8 = np.uint8(np.clip(expected * 35 + 128, 0, 255))

    center_edges = cv2.Canny(center_u8, 30, 80)
    expected_edges = cv2.Canny(expected_u8, 30, 80)

    edge_difference = float(
        np.mean(
            np.abs(
                center_edges.astype(np.float32)
                - expected_edges.astype(np.float32)
            )
        ) / 255.0
    )

    score = (
        0.35 * mean_residual
        + 0.40 * high_residual
        + 0.10 * strong_fraction
        + 0.15 * edge_difference
    )

    return {
        "anomaly": float(score),
        "mean_residual": mean_residual,
        "high_residual": high_residual,
        "edge_difference": edge_difference,
    }


def remove_duplicates(candidates, distance=20):
    candidates = sorted(
        candidates,
        key=lambda z: z["final_score"],
        reverse=True,
    )

    result = []

    for candidate in candidates:
        keep = True

        for previous in result:
            d = np.hypot(
                candidate["x"] - previous["x"],
                candidate["y"] - previous["y"],
            )

            if d < distance:
                keep = False
                break

        if keep:
            result.append(candidate)

    return result


def localize(reference, search):
    print("Estimating DRAM periodicity...")

    pitch_x, pitch_y = estimate_pitch(search)

    print(
        f"Estimated pitch: x={pitch_x:.2f}, y={pitch_y:.2f}"
    )

    proposals = []

    print("Generating template proposals...")

    for scale in SCALES:
        for rotation in ROTATIONS:
            template = make_template(
                reference,
                float(scale),
                float(rotation),
            )

            score = template_score(search, template)
            if score is None:
                continue

            peaks = extract_peaks(
                score,
                TOP_TEMPLATE_PEAKS,
                suppression=12,
            )

            h, w = template.shape

            for px, py, raw in peaks:
                cx = px + w / 2.0
                cy = py + h / 2.0

                if (
                    cx < PATCH_SIZE / 2
                    or cy < PATCH_SIZE / 2
                    or cx > search.shape[1] - PATCH_SIZE / 2
                    or cy > search.shape[0] - PATCH_SIZE / 2
                ):
                    continue

                proposals.append(
                    {
                        "x": cx,
                        "y": cy,
                        "raw": float(raw),
                        "scale": float(scale),
                        "rotation": float(rotation),
                    }
                )

    print(f"Generated {len(proposals)} raw proposals.")

    proposals.sort(key=lambda z: z["raw"], reverse=True)

    # Search a lattice around strong template locations.
    seeds = proposals[:80]
    dense = []

    for seed in seeds:
        sx = seed["x"]
        sy = seed["y"]

        for ix in range(-8, 9):
            for iy in range(-8, 9):
                cx = sx + ix * pitch_x
                cy = sy + iy * pitch_y

                if (
                    cx < PATCH_SIZE / 2
                    or cy < PATCH_SIZE / 2
                    or cx > search.shape[1] - PATCH_SIZE / 2
                    or cy > search.shape[0] - PATCH_SIZE / 2
                ):
                    continue

                dense.append(
                    {
                        "x": float(cx),
                        "y": float(cy),
                        "raw": seed["raw"],
                        "scale": seed["scale"],
                        "rotation": seed["rotation"],
                    }
                )

    proposals.extend(dense)

    unique = []
    proposals.sort(key=lambda z: z["raw"], reverse=True)

    for p in proposals:
        if all(
            np.hypot(p["x"] - q["x"], p["y"] - q["y"]) > 10
            for q in unique
        ):
            unique.append(p)

        if len(unique) >= 2500:
            break

    print(f"Testing {len(unique)} unique candidates...")

    results = []

    for candidate in unique:
        anomaly = periodic_anomaly(
            search,
            candidate["x"],
            candidate["y"],
            pitch_x,
            pitch_y,
        )

        if anomaly is None:
            continue

        candidate = dict(candidate)
        candidate.update(anomaly)
        results.append(candidate)

    if not results:
        raise RuntimeError("No valid periodic candidates.")

    anomalies = np.array(
        [r["anomaly"] for r in results],
        dtype=np.float32,
    )

    raws = np.array(
        [r["raw"] for r in results],
        dtype=np.float32,
    )

    def robust_normalize(values):
        lo = np.percentile(values, 5)
        hi = np.percentile(values, 95)

        if hi <= lo:
            return np.zeros_like(values)

        return np.clip(
            (values - lo) / (hi - lo),
            0,
            1,
        )

    anomaly_norm = robust_normalize(anomalies)
    raw_norm = robust_normalize(raws)

    for i, r in enumerate(results):
        r["anomaly_norm"] = float(anomaly_norm[i])
        r["raw_norm"] = float(raw_norm[i])

        r["final_score"] = (
            0.82 * r["anomaly_norm"]
            + 0.18 * r["raw_norm"]
        )

    results = remove_duplicates(results, distance=25)
    results.sort(key=lambda z: z["final_score"], reverse=True)

    return results[:20], (pitch_x, pitch_y)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--gt", default=None)

    args = parser.parse_args()

    print()
    print("=" * 64)
    print("PERIODICITY-AWARE DRAM LOCALIZATION")
    print("=" * 64)

    reference = load_gray(args.reference)
    search = load_gray(args.search)

    print(
        f"Reference: {reference.shape[1]} x {reference.shape[0]}"
    )
    print(
        f"Search: {search.shape[1]} x {search.shape[0]}"
    )

    results, pitch = localize(reference, search)

    print()
    print("=" * 64)
    print("TOP PERIODICITY-AWARE CANDIDATES")
    print("=" * 64)

    for i, r in enumerate(results, 1):
        print(
            f"{i:02d}. "
            f"x={r['x']:.2f} "
            f"y={r['y']:.2f} "
            f"score={r['final_score']:.5f} "
            f"anomaly={r['anomaly']:.5f} "
            f"raw={r['raw']:.5f}"
        )

    best = results[0]

    print()
    print("=" * 64)
    print("FINAL DRAM LOCALIZATION")
    print("=" * 64)

    print(f"x                 : {best['x']:.3f}")
    print(f"y                 : {best['y']:.3f}")
    print(f"final score       : {best['final_score']:.5f}")
    print(f"periodic anomaly  : {best['anomaly']:.5f}")
    print(f"raw similarity    : {best['raw']:.5f}")
    print(f"scale             : {best['scale']:.5f}")
    print(f"rotation          : {best['rotation']:.2f} deg")
    print(f"pitch             : {pitch[0]:.2f}, {pitch[1]:.2f}")

    if args.gt is not None:
        gt_path = Path(args.gt)

        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)

            target = gt["target_center_search"]
            gx = float(target["x"])
            gy = float(target["y"])

            error = np.hypot(
                best["x"] - gx,
                best["y"] - gy,
            )

            print()
            print("GROUND TRUTH CHECK")
            print(f"GT x,y            : {gx:.3f}, {gy:.3f}")
            print(f"Localization error: {error:.3f} px")

    print("=" * 64)


if __name__ == "__main__":
    main()
