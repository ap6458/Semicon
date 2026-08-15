import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("dataset")

TRAIN_END = 400
VAL_END = 450

PATCH_SIZE = 128
INPUT_SIZE = 96

# FAST TRAINING SETTINGS
EPOCHS = 5
TRAIN_EXAMPLES_PER_SAMPLE = 8
VAL_EXAMPLES_PER_SAMPLE = 10

BATCH_SIZE = 32
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_gray(path):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None or img.size == 0:
        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return img


def read_gt(sample_dir):

    with open(
        sample_dir / "ground_truth.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def crop_center(
    img,
    cx,
    cy,
    size
):

    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    half = size // 2

    cx = int(round(cx))
    cy = int(round(cy))

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size

    if x0 < 0 or y0 < 0:
        return None

    if x1 > w or y1 > h:
        return None

    patch = img[
        y0:y1,
        x0:x1
    ]

    if patch.shape != (
        size,
        size
    ):
        return None

    return np.ascontiguousarray(
        patch,
        dtype=np.uint8
    )


def preprocess(img):

    if img is None or img.size == 0:
        raise RuntimeError(
            "preprocess() received empty image"
        )

    img = cv2.resize(
        img,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA
    )

    img = (
        img.astype(np.float32)
        / 255.0
    )

    mean = img.mean()
    std = img.std() + 1e-6

    img = (
        img - mean
    ) / std

    return img.astype(
        np.float32
    )


# ============================================================
# REFERENCE TRANSFORMATION
# ============================================================

def transform_reference(
    reference,
    gt
):

    scale_factor = float(
        gt["scale_factor"]
    )

    scale = 1.0 / scale_factor

    h, w = reference.shape

    nw = max(
        16,
        int(round(w * scale))
    )

    nh = max(
        16,
        int(round(h * scale))
    )

    small = cv2.resize(
        reference,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    rotation = float(
        gt.get(
            "reference_rotation_deg",
            0.0
        )
    )

    if abs(rotation) > 0.01:

        matrix = cv2.getRotationMatrix2D(
            (
                small.shape[1] / 2,
                small.shape[0] / 2
            ),
            rotation,
            1.0
        )

        small = cv2.warpAffine(
            small,
            matrix,
            (
                small.shape[1],
                small.shape[0]
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    canvas = np.zeros(
        (
            PATCH_SIZE,
            PATCH_SIZE
        ),
        dtype=np.uint8
    )

    if (
        small.shape[0] > PATCH_SIZE
        or
        small.shape[1] > PATCH_SIZE
    ):

        small = cv2.resize(
            small,
            (
                PATCH_SIZE,
                PATCH_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        return small

    y0 = (
        PATCH_SIZE - small.shape[0]
    ) // 2

    x0 = (
        PATCH_SIZE - small.shape[1]
    ) // 2

    canvas[
        y0:y0 + small.shape[0],
        x0:x0 + small.shape[1]
    ] = small

    return canvas


# ============================================================
# HARD NEGATIVE GENERATION
# ============================================================

def find_hard_negatives(
    search,
    reference_patch,
    target_x,
    target_y
):

    # Use a smaller representation for proposal generation
    # to reduce CPU time.
    search_small = cv2.resize(
        search,
        (500, 500),
        interpolation=cv2.INTER_AREA
    )

    reference_small = cv2.resize(
        reference_patch,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    s = search_small.astype(
        np.float32
    )

    t = reference_small.astype(
        np.float32
    )

    s = (
        s - s.mean()
    ) / (s.std() + 1e-6)

    t = (
        t - t.mean()
    ) / (t.std() + 1e-6)

    result = cv2.matchTemplate(
        s,
        t,
        cv2.TM_CCOEFF_NORMED
    )

    locations = []

    for _ in range(15):

        _, score, _, loc = (
            cv2.minMaxLoc(result)
        )

        if score < 0.05:
            break

        x, y = loc

        # Convert 500x500 coordinates back to
        # the original 1000x1000 search coordinates.
        cx = (
            x + 32
        ) * 2.0

        cy = (
            y + 32
        ) * 2.0

        distance = np.hypot(
            cx - target_x,
            cy - target_y
        )

        if distance > 80:

            locations.append(
                (
                    float(cx),
                    float(cy),
                    float(score)
                )
            )

        r = 20

        x0 = max(
            0,
            x - r
        )

        x1 = min(
            result.shape[1],
            x + r + 1
        )

        y0 = max(
            0,
            y - r
        )

        y1 = min(
            result.shape[0],
            y + r + 1
        )

        result[
            y0:y1,
            x0:x1
        ] = -np.inf

    return locations


# ============================================================
# DATASET
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        sample_dirs,
        examples_per_sample
    ):

        self.samples = sample_dirs

        self.examples_per_sample = (
            examples_per_sample
        )

    def __len__(self):

        return (
            len(self.samples)
            *
            self.examples_per_sample
        )

    def __getitem__(self, index):

        sample_index = (
            index
            //
            self.examples_per_sample
        )

        sample = self.samples[
            sample_index
        ]

        for _ in range(20):

            try:

                gt = read_gt(sample)

                reference = load_gray(
                    sample / "reference.png"
                )

                search = load_gray(
                    sample / "search.png"
                )

                target = gt[
                    "target_center_search"
                ]

                cx = float(
                    target["x"]
                )

                cy = float(
                    target["y"]
                )

                reference_patch = (
                    transform_reference(
                        reference,
                        gt
                    )
                )

                # --------------------------------------------
                # POSITIVE
                # --------------------------------------------

                if random.random() < 0.5:

                    px = (
                        cx
                        +
                        random.uniform(
                            -6,
                            6
                        )
                    )

                    py = (
                        cy
                        +
                        random.uniform(
                            -6,
                            6
                        )
                    )

                    label = 1.0

                # --------------------------------------------
                # HARD NEGATIVE
                # --------------------------------------------

                else:

                    hard = find_hard_negatives(
                        search,
                        reference_patch,
                        cx,
                        cy
                    )

                    if hard:

                        px, py, _ = random.choice(
                            hard
                        )

                    else:

                        h, w = search.shape

                        px = random.uniform(
                            PATCH_SIZE / 2,
                            w - PATCH_SIZE / 2
                        )

                        py = random.uniform(
                            PATCH_SIZE / 2,
                            h - PATCH_SIZE / 2
                        )

                    label = 0.0

                search_patch = crop_center(
                    search,
                    px,
                    py,
                    PATCH_SIZE
                )

                if search_patch is None:
                    continue

                ref = preprocess(
                    reference_patch
                )

                candidate = preprocess(
                    search_patch
                )

                x = np.stack(
                    [
                        ref,
                        candidate
                    ],
                    axis=0
                )

                return (
                    torch.tensor(
                        x,
                        dtype=torch.float32
                    ),
                    torch.tensor(
                        label,
                        dtype=torch.float32
                    )
                )

            except Exception:
                continue

        raise RuntimeError(
            f"Could not create sample {index}"
        )


# ============================================================
# MODEL
# ============================================================

class MatchNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                2,
                16,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(
            x
        ).squeeze(1)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            correct += (
                predictions == y
            ).sum().item()

            total += y.numel()

            total_loss += (
                loss.item()
                *
                y.numel()
            )

    return (
        total_loss / max(total, 1),
        100.0 * correct / max(total, 1)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    samples = sorted(
        [
            p
            for p in DATASET_ROOT.glob(
                "sample_*"
            )
            if p.is_dir()
        ]
    )

    print(
        f"Found {len(samples)} samples."
    )

    if len(samples) < 450:

        raise RuntimeError(
            "Need at least 450 samples."
        )

    train_samples = samples[
        :TRAIN_END
    ]

    val_samples = samples[
        TRAIN_END:VAL_END
    ]

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Train samples: "
        f"{len(train_samples)} "
        f"-> "
        f"{len(train_samples) * TRAIN_EXAMPLES_PER_SAMPLE} pairs"
    )

    print(
        f"Val samples: "
        f"{len(val_samples)} "
        f"-> "
        f"{len(val_samples) * VAL_EXAMPLES_PER_SAMPLE} pairs"
    )

    train_dataset = PairDataset(
        train_samples,
        examples_per_sample=TRAIN_EXAMPLES_PER_SAMPLE
    )

    val_dataset = PairDataset(
        val_samples,
        examples_per_sample=VAL_EXAMPLES_PER_SAMPLE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = MatchNet().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_accuracy = 0.0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        total = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                *
                y.numel()
            )

            total += y.numel()

        train_loss = (
            running_loss
            /
            max(total, 1)
        )

        val_loss, val_accuracy = (
            evaluate(
                model,
                val_loader
            )
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "match_ranker.pt"
            )

            print(
                "  saved match_ranker.pt"
            )

    print()
    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()