import argparse
import cv2
import numpy as np
import torch

from train_ranker import (
    MatchNet,
    PATCH_SIZE,
    preprocess,
    load_gray,
    DEVICE,
)

# ============================================================
# SETTINGS
# ============================================================

SCALES = np.arange(0.0925, 0.1081, 0.0025)
ROTATIONS = np.arange(-3.0, 3.01, 1.0)

TOP_PER_HYPOTHESIS = 40
MAX_CANDIDATES = 500
BATCH = 128


# ============================================================
# TEMPLATE
# ============================================================

def make_template(reference, scale, rotation):
    h, w = reference.shape

    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))

    template = cv2.resize(
        reference,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    M = cv2.getRotationMatrix2D(
        (nw / 2.0, nh / 2.0),
        rotation,
        1.0
    )

    template = cv2.warpAffine(
        template,
        M,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return np.ascontiguousarray(
        template,
        dtype=np.uint8
    )


# ============================================================
# NORMALIZATION
# ============================================================

def norm(img):
    img = np.asarray(
        img,
        dtype=np.float32
    )

    img = np.squeeze(img)

    if img.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D image, got {img.shape}"
        )

    lo, hi = np.percentile(
        img,
        [1, 99]
    )

    img = np.clip(
        (img - lo) / (hi - lo + 1e-6),
        0.0,
        1.0
    )

    return np.ascontiguousarray(
        img,
        dtype=np.float32
    )


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def raw_score_map(search, template):

    search_n = norm(search)
    template_n = norm(template)

    if template_n.shape[0] > search_n.shape[0]:
        return None

    if template_n.shape[1] > search_n.shape[1]:
        return None

    return cv2.matchTemplate(
        search_n,
        template_n,
        cv2.TM_CCOEFF_NORMED
    )


# ============================================================
# PEAK DETECTION
# ============================================================

def peaks(score, k):

    work = score.copy()
    result = []

    for _ in range(k):

        _, value, _, location = cv2.minMaxLoc(work)

        if value < -0.99:
            break

        x, y = location

        result.append(
            (x, y, float(value))
        )

        r = 10

        x0 = max(0, x - r)
        x1 = min(work.shape[1], x + r + 1)

        y0 = max(0, y - r)
        y1 = min(work.shape[0], y + r + 1)

        work[y0:y1, x0:x1] = -np.inf

    return result


# ============================================================
# SAFE CROP
# ============================================================

def crop(img, cx, cy, size=PATCH_SIZE):

    half = size // 2

    x = int(round(cx))
    y = int(round(cy))

    x0 = x - half
    y0 = y - half

    x1 = x0 + size
    y1 = y0 + size

    if x0 < 0 or y0 < 0:
        return None

    if x1 > img.shape[1] or y1 > img.shape[0]:
        return None

    patch = img[y0:y1, x0:x1]

    if patch.shape != (size, size):
        return None

    return np.ascontiguousarray(
        patch,
        dtype=np.uint8
    )


# ============================================================
# REFERENCE PATCH
# ============================================================

def make_reference_patch(template):

    canvas = np.zeros(
        (PATCH_SIZE, PATCH_SIZE),
        dtype=np.uint8
    )

    h, w = template.shape

    if h > PATCH_SIZE or w > PATCH_SIZE:

        scale = min(
            PATCH_SIZE / w,
            PATCH_SIZE / h
        )

        new_w = max(
            16,
            int(round(w * scale))
        )

        new_h = max(
            16,
            int(round(h * scale))
        )

        template = cv2.resize(
            template,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        h, w = template.shape

    x0 = (PATCH_SIZE - w) // 2
    y0 = (PATCH_SIZE - h) // 2

    canvas[
        y0:y0+h,
        x0:x0+w
    ] = template

    return canvas


# ============================================================
# PREPROCESS BATCH
# ============================================================

def preprocess_pair(ref_patch, cand_patch):

    ref = preprocess(ref_patch)
    cand = preprocess(cand_patch)

    if not torch.is_tensor(ref):
        ref = torch.tensor(
            ref,
            dtype=torch.float32
        )

    if not torch.is_tensor(cand):
        cand = torch.tensor(
            cand,
            dtype=torch.float32
        )

    # Ensure [C,H,W]
    if ref.ndim == 2:
        ref = ref.unsqueeze(0)

    if cand.ndim == 2:
        cand = cand.unsqueeze(0)

    return ref, cand


# ============================================================
# MODEL FORWARD
# ============================================================

def run_model(model, refs, cands):

    # First try the common two-input form.
    try:
        return model(refs, cands)
    except TypeError:
        pass

    # Otherwise use the combined [B,2,H,W] form.
    pair = torch.cat(
        [refs, cands],
        dim=1
    )

    return model(pair)


# ============================================================
# LOCALIZATION
# ============================================================

def localize(reference_path, search_path, model_path):

    print()
    print("Loading images...")

    reference = load_gray(reference_path)
    search = load_gray(search_path)

    reference = np.asarray(
        reference,
        dtype=np.uint8
    )

    search = np.asarray(
        search,
        dtype=np.uint8
    )

    reference = np.squeeze(reference)
    search = np.squeeze(search)

    print(
        f"Reference: "
        f"{reference.shape[1]} x {reference.shape[0]}"
    )

    print(
        f"Search: "
        f"{search.shape[1]} x {search.shape[0]}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("Loading model...")

    model = MatchNet().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Your train_ranker.py saves state_dict directly.
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    print("Model loaded.")

    # --------------------------------------------------------
    # PROPOSALS
    # --------------------------------------------------------

    print()
    print(
        f"Generating proposals "
        f"({len(SCALES) * len(ROTATIONS)} "
        f"scale/rotation hypotheses)..."
    )

    proposals = []

    for scale in SCALES:

        for rotation in ROTATIONS:

            template = make_template(
                reference,
                float(scale),
                float(rotation)
            )

            score = raw_score_map(
                search,
                template
            )

            if score is None:
                continue

            for x, y, raw in peaks(
                score,
                TOP_PER_HYPOTHESIS
            ):

                h, w = template.shape

                cx = x + w / 2.0
                cy = y + h / 2.0

                candidate_patch = crop(
                    search,
                    cx,
                    cy
                )

                if candidate_patch is None:
                    continue

                proposals.append({
                    "x": float(cx),
                    "y": float(cy),
                    "raw": float(raw),
                    "scale": float(scale),
                    "rotation": float(rotation),
                    "candidate": candidate_patch,
                    "template": template
                })

    print(
        f"Raw proposals: {len(proposals)}"
    )

    if not proposals:
        raise RuntimeError(
            "No valid proposals generated."
        )

    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    proposals.sort(
        key=lambda p: p["raw"],
        reverse=True
    )

    unique = []

    for p in proposals:

        duplicate = False

        for q in unique:

            distance = np.hypot(
                p["x"] - q["x"],
                p["y"] - q["y"]
            )

            if distance <= 12:
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

        if len(unique) >= MAX_CANDIDATES:
            break

    print(
        f"Unique candidates: {len(unique)}"
    )

    # --------------------------------------------------------
    # CNN RANKING
    # --------------------------------------------------------

    print(
        "Ranking candidates with CNN..."
    )

    ranked = []

    with torch.no_grad():

        for start in range(
            0,
            len(unique),
            BATCH
        ):

            batch = unique[
                start:start + BATCH
            ]

            refs = []
            cands = []

            for p in batch:

                ref_patch = make_reference_patch(
                    p["template"]
                )

                ref, cand = preprocess_pair(
                    ref_patch,
                    p["candidate"]
                )

                refs.append(ref)
                cands.append(cand)

            refs = torch.stack(refs).to(DEVICE)
            cands = torch.stack(cands).to(DEVICE)

            logits = run_model(
                model,
                refs,
                cands
            )

            logits = logits.reshape(-1)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            for p, probability in zip(
                batch,
                probabilities
            ):

                ranked.append({
                    "x": p["x"],
                    "y": p["y"],
                    "probability": float(
                        probability
                    ),
                    "raw_score": p["raw"],
                    "scale": p["scale"],
                    "rotation": p["rotation"]
                })

    ranked.sort(
        key=lambda p: p["probability"],
        reverse=True
    )

    best = ranked[0]

    return {
        "x": best["x"],
        "y": best["y"],
        "probability": best["probability"],
        "raw_score": best["raw_score"],
        "scale": best["scale"],
        "rotation": best["rotation"],
        "num_candidates": len(ranked),
        "top_candidates": ranked[:10]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        required=True
    )

    parser.add_argument(
        "--search",
        required=True
    )

    parser.add_argument(
        "--model",
        default="match_ranker.pt"
    )

    args = parser.parse_args()

    result = localize(
        args.reference,
        args.search,
        args.model
    )

    print()
    print("=" * 65)
    print("ML DRAM LOCALIZATION")
    print("=" * 65)

    print(
        f"x             : "
        f"{result['x']:.3f}"
    )

    print(
        f"y             : "
        f"{result['y']:.3f}"
    )

    print(
        f"probability   : "
        f"{result['probability']:.5f}"
    )

    print(
        f"raw proposal  : "
        f"{result['raw_score']:.5f}"
    )

    print(
        f"scale         : "
        f"{result['scale']:.5f}"
    )

    print(
        f"rotation      : "
        f"{result['rotation']:.2f} deg"
    )

    print(
        f"candidates    : "
        f"{result['num_candidates']}"
    )

    print("=" * 65)

    print()
    print("TOP 10 CANDIDATES")
    print("-" * 65)

    for i, p in enumerate(
        result["top_candidates"],
        1
    ):

        print(
            f"{i:02d}. "
            f"x={p['x']:.2f} "
            f"y={p['y']:.2f} "
            f"prob={p['probability']:.4f} "
            f"raw={p['raw_score']:.4f}"
        )


if __name__ == "__main__":
    main()