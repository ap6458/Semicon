import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_NUM_PAIRS = 500
DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_SEED = 42

HIGH_RES_SIZE = 12000

REFERENCE_SIZE = 1000
SEARCH_SIZE = 1000

# Required approximately 10x relationship.
# We intentionally vary it around 10x.
MIN_SCALE = 9.5
MAX_SCALE = 10.5


# ============================================================
# RANDOMNESS
# ============================================================

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_uint8(image):
    low, high = np.percentile(image, [1, 99])

    image = (image - low) / (high - low + 1e-8)
    image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


# ============================================================
# DRAM BASE LAYOUT
# ============================================================

def generate_dram_layout(size):
    """
    Generates a highly periodic DRAM-style layout.

    Structure:
        horizontal word lines
        vertical bit lines
        contact/via structures
        slight manufacturing variation
        sparse missing contacts

    The layout remains strongly periodic because that is the
    fundamental challenge of the hackathon.
    """

    image = np.full(
        (size, size),
        random.uniform(25, 55),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Global DRAM parameters
    # --------------------------------------------------------

    horizontal_pitch = random.uniform(
        80,
        125
    )

    vertical_pitch = random.uniform(
        80,
        125
    )

    horizontal_width = random.uniform(
        6,
        13
    )

    vertical_width = random.uniform(
        6,
        13
    )

    # Slight anisotropy makes layouts less artificial.
    horizontal_pitch *= random.uniform(
        0.98,
        1.02
    )

    vertical_pitch *= random.uniform(
        0.98,
        1.02
    )

    # --------------------------------------------------------
    # Horizontal word lines
    # --------------------------------------------------------

    y_positions = []

    y = random.uniform(
        0,
        horizontal_pitch
    )

    while y < size:

        y_positions.append(
            int(round(y))
        )

        width = max(
            2,
            int(round(
                horizontal_width
                *
                random.uniform(
                    0.90,
                    1.10
                )
            ))
        )

        intensity = random.uniform(
            135,
            205
        )

        y0 = max(
            0,
            int(y - width / 2)
        )

        y1 = min(
            size,
            int(y + width / 2 + 1)
        )

        image[y0:y1, :] = intensity

        y += (
            horizontal_pitch
            *
            random.uniform(
                0.985,
                1.015
            )
        )

    # --------------------------------------------------------
    # Vertical bit lines
    # --------------------------------------------------------

    x_positions = []

    x = random.uniform(
        0,
        vertical_pitch
    )

    while x < size:

        x_positions.append(
            int(round(x))
        )

        width = max(
            2,
            int(round(
                vertical_width
                *
                random.uniform(
                    0.90,
                    1.10
                )
            ))
        )

        intensity = random.uniform(
            135,
            205
        )

        x0 = max(
            0,
            int(x - width / 2)
        )

        x1 = min(
            size,
            int(x + width / 2 + 1)
        )

        image[:, x0:x1] = intensity

        x += (
            vertical_pitch
            *
            random.uniform(
                0.985,
                1.015
            )
        )

    # --------------------------------------------------------
    # Contacts / vias
    # --------------------------------------------------------

    contact_radius = random.uniform(
        5,
        10
    )

    for y in y_positions:

        for x in x_positions:

            # Sparse missing contacts create realistic
            # non-periodic information.

            if random.random() < 0.004:
                continue

            radius = max(
                2,
                int(round(
                    contact_radius
                    *
                    random.uniform(
                        0.85,
                        1.15
                    )
                ))
            )

            intensity = random.uniform(
                205,
                255
            )

            cv2.circle(
                image,
                (x, y),
                radius,
                intensity,
                -1
            )

    return image


# ============================================================
# TARGET-LOCAL STRUCTURAL VARIATION
# ============================================================

def add_subtle_target_features(
    image,
    target_center_x,
    target_center_y
):
    """
    Adds subtle, semiconductor-like local variations around
    the target.

    IMPORTANT:
    These are intentionally NOT giant artificial blocks.

    They represent small local structural differences such as:
        - missing contact
        - extra contact
        - line interruption
        - local pitch perturbation
        - small bridge
        - local line-width variation
    """

    h, w = image.shape

    cx = int(target_center_x)
    cy = int(target_center_y)

    records = []

    # --------------------------------------------------------
    # Feature 1: missing/attenuated contact
    # --------------------------------------------------------

    dx = random.randint(
        -250,
        250
    )

    dy = random.randint(
        -250,
        250
    )

    fx = np.clip(
        cx + dx,
        80,
        w - 80
    )

    fy = np.clip(
        cy + dy,
        80,
        h - 80
    )

    radius = random.randint(
        12,
        24
    )

    y0 = max(
        0,
        fy - radius
    )

    y1 = min(
        h,
        fy + radius + 1
    )

    x0 = max(
        0,
        fx - radius
    )

    x1 = min(
        w,
        fx + radius + 1
    )

    yy, xx = np.ogrid[
        y0:y1,
        x0:x1
    ]

    mask = (
        (yy - fy) ** 2
        +
        (xx - fx) ** 2
        <= radius ** 2
    )

    image[y0:y1, x0:x1][mask] *= random.uniform(
        0.15,
        0.40
    )

    records.append({
        "type": "missing_contact",
        "x_high_res": int(fx),
        "y_high_res": int(fy)
    })

    # --------------------------------------------------------
    # Feature 2: local line interruption
    # --------------------------------------------------------

    fx = np.clip(
        cx + random.randint(-300, 300),
        100,
        w - 100
    )

    fy = np.clip(
        cy + random.randint(-300, 300),
        100,
        h - 100
    )

    length = random.randint(
        50,
        100
    )

    thickness = random.randint(
        5,
        12
    )

    # Remove a small horizontal section.
    image[
        fy - thickness:
        fy + thickness + 1,

        fx - length:
        fx + length + 1
    ] *= random.uniform(
        0.25,
        0.55
    )

    records.append({
        "type": "line_interruption",
        "x_high_res": int(fx),
        "y_high_res": int(fy)
    })

    # --------------------------------------------------------
    # Feature 3: small bridge / local connection
    # --------------------------------------------------------

    fx = np.clip(
        cx + random.randint(-300, 300),
        100,
        w - 100
    )

    fy = np.clip(
        cy + random.randint(-300, 300),
        100,
        h - 100
    )

    length = random.randint(
        50,
        100
    )

    thickness = random.randint(
        6,
        14
    )

    cv2.rectangle(
        image,
        (
            fx - length // 2,
            fy - thickness // 2
        ),
        (
            fx + length // 2,
            fy + thickness // 2
        ),
        random.uniform(
            170,
            220
        ),
        -1
    )

    records.append({
        "type": "local_bridge",
        "x_high_res": int(fx),
        "y_high_res": int(fy)
    })

    # --------------------------------------------------------
    # Feature 4: subtle local pitch perturbation
    # --------------------------------------------------------

    fx = np.clip(
        cx + random.randint(-300, 300),
        150,
        w - 150
    )

    fy = np.clip(
        cy + random.randint(-300, 300),
        150,
        h - 150
    )

    patch_size = random.randint(
        100,
        180
    )

    patch_x0 = fx - patch_size // 2
    patch_x1 = fx + patch_size // 2

    patch_y0 = fy - patch_size // 2
    patch_y1 = fy + patch_size // 2

    patch = image[
        patch_y0:patch_y1,
        patch_x0:patch_x1
    ].copy()

    # Slight local scaling introduces a small structural
    # perturbation rather than a completely new object.

    local_scale = random.uniform(
        0.94,
        1.06
    )

    new_size = int(
        patch_size * local_scale
    )

    if new_size > 10:

        resized = cv2.resize(
            patch,
            (new_size, new_size),
            interpolation=cv2.INTER_LINEAR
        )

        center = new_size // 2

        half = patch_size // 2

        crop = resized[
            center - half:
            center + half,

            center - half:
            center + half
        ]

        if crop.shape == patch.shape:

            image[
                patch_y0:patch_y1,
                patch_x0:patch_x1
            ] = crop

    records.append({
        "type": "local_pitch_perturbation",
        "x_high_res": int(fx),
        "y_high_res": int(fy),
        "scale": float(local_scale)
    })

    return image, records


# ============================================================
# SEM EDGE BRIGHTENING
# ============================================================

def apply_edge_brightening(
    image,
    strength
):
    """
    Approximate SEM edge enhancement.

    High-frequency structure is extracted using an unsharp-mask
    style operation and added back to the image.
    """

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=random.uniform(
            0.8,
            2.0
        )
    )

    high_frequency = (
        image - blurred
    )

    result = (
        image
        +
        strength * high_frequency
    )

    return np.clip(
        result,
        0,
        255
    )


# ============================================================
# BLUR
# ============================================================

def apply_blur(
    image,
    sigma_range
):

    sigma = random.uniform(
        sigma_range[0],
        sigma_range[1]
    )

    if sigma <= 0:
        return image

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma
    )


# ============================================================
# GAUSSIAN SENSOR NOISE
# ============================================================

def add_gaussian_noise(
    image,
    sigma_range
):

    sigma = random.uniform(
        sigma_range[0],
        sigma_range[1]
    )

    # IMPORTANT:
    # np.random.normal creates a NEW noise realization every
    # time this function is called.

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(
        np.float32
    )

    return np.clip(
        image + noise,
        0,
        255
    )


# ============================================================
# POISSON / SHOT NOISE
# ============================================================

def add_poisson_noise(
    image,
    strength_range
):

    strength = random.uniform(
        strength_range[0],
        strength_range[1]
    )

    normalized = np.clip(
        image / 255.0,
        0,
        1
    )

    noisy = np.random.poisson(
        normalized * strength
    ) / strength

    return np.clip(
        noisy * 255,
        0,
        255
    )


# ============================================================
# BRIGHTNESS / CONTRAST
# ============================================================

def change_brightness_contrast(
    image,
    brightness_range,
    contrast_range
):

    brightness = random.uniform(
        brightness_range[0],
        brightness_range[1]
    )

    contrast = random.uniform(
        contrast_range[0],
        contrast_range[1]
    )

    mean = np.mean(image)

    result = (
        (image - mean)
        * contrast
        +
        mean
        +
        brightness
    )

    return np.clip(
        result,
        0,
        255
    )


# ============================================================
# REFERENCE CAPTURE
# ============================================================

def augment_reference(
    reference
):

    # Reference is intentionally cleaner.

    result = apply_edge_brightening(
        reference,
        strength=random.uniform(
            0.20,
            0.45
        )
    )

    result = apply_blur(
        result,
        (
            0.0,
            0.45
        )
    )

    result = change_brightness_contrast(
        result,
        (-5, 5),
        (0.95, 1.08)
    )

    # Independent sensor noise.
    result = add_gaussian_noise(
        result,
        (1.5, 5.0)
    )

    if random.random() < 0.35:

        result = add_poisson_noise(
            result,
            (100, 200)
        )

    # --------------------------------------------------------
    # Small reference rotation
    # --------------------------------------------------------

    angle = random.uniform(
        -2.0,
        2.0
    )

    h, w = result.shape

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    result = cv2.warpAffine(
        result,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # --------------------------------------------------------
    # Small independent reference scale variation
    # --------------------------------------------------------

    scale = random.uniform(
        0.98,
        1.02
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    scaled = cv2.resize(
        result,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    if scale >= 1.0:

        start_x = (
            new_w - w
        ) // 2

        start_y = (
            new_h - h
        ) // 2

        result = scaled[
            start_y:start_y + h,
            start_x:start_x + w
        ]

    else:

        result = np.pad(
            scaled,
            (
                (
                    (h - new_h) // 2,
                    h - new_h - (h - new_h) // 2
                ),
                (
                    (w - new_w) // 2,
                    w - new_w - (w - new_w) // 2
                )
            ),
            mode="reflect"
        )

    return (
        np.clip(
            result,
            0,
            255
        ),
        angle,
        scale
    )


# ============================================================
# SEARCH CAPTURE
# ============================================================

def augment_search(
    search
):

    # Search is deliberately noisier.

    result = apply_edge_brightening(
        search,
        strength=random.uniform(
            0.30,
            0.65
        )
    )

    result = apply_blur(
        result,
        (
            0.1,
            0.8
        )
    )

    result = change_brightness_contrast(
        result,
        (-12, 12),
        (0.80, 1.15)
    )

    # IMPORTANT:
    # Completely independent noise realization.

    result = add_gaussian_noise(
        result,
        (7.0, 18.0)
    )

    if random.random() < 0.70:

        result = add_poisson_noise(
            result,
            (40, 100)
        )

    return np.clip(
        result,
        0,
        255
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    path,
    image
):

    image = normalize_uint8(
        image
    )

    cv2.imwrite(
        str(path),
        image
    )


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_sample(
    sample_id,
    output_dir
):

    # --------------------------------------------------------
    # Generate a large physical DRAM layout.
    # --------------------------------------------------------

    layout = generate_dram_layout(
        HIGH_RES_SIZE
    )

    layout_h, layout_w = layout.shape

    # --------------------------------------------------------
    # Random search magnification.
    #
    # 9.5x means:
    #
    # 1000 high-res reference pixels
    # appear as ~105.3 search pixels.
    #
    # 10.5x means:
    #
    # 1000 high-res reference pixels
    # appear as ~95.2 search pixels.
    # --------------------------------------------------------

    scale_factor = random.uniform(
        MIN_SCALE,
        MAX_SCALE
    )

    search_physical_size = int(
        round(
            SEARCH_SIZE
            *
            scale_factor
        )
    )

    # --------------------------------------------------------
    # Select a physical search crop.
    # --------------------------------------------------------

    max_crop_x = (
        layout_w
        -
        search_physical_size
    )

    max_crop_y = (
        layout_h
        -
        search_physical_size
    )

    if (
        max_crop_x <= 0
        or
        max_crop_y <= 0
    ):

        raise RuntimeError(
            "HIGH_RES_SIZE is too small for "
            "the selected scale factor."
        )

    search_origin_x = random.randint(
        0,
        max_crop_x
    )

    search_origin_y = random.randint(
        0,
        max_crop_y
    )

    # --------------------------------------------------------
    # Choose target center inside search crop.
    #
    # Keep it away from the borders so that the 1000x1000
    # reference crop is fully contained in the physical layout.
    # --------------------------------------------------------

    min_target_x = (
        search_origin_x
        +
        REFERENCE_SIZE // 2
    )

    max_target_x = (
        search_origin_x
        +
        search_physical_size
        -
        REFERENCE_SIZE // 2
    )

    min_target_y = (
        search_origin_y
        +
        REFERENCE_SIZE // 2
    )

    max_target_y = (
        search_origin_y
        +
        search_physical_size
        -
        REFERENCE_SIZE // 2
    )

    target_center_x = random.randint(
        min_target_x,
        max_target_x
    )

    target_center_y = random.randint(
        min_target_y,
        max_target_y
    )

    # --------------------------------------------------------
    # Reference crop position.
    # --------------------------------------------------------

    reference_x = (
        target_center_x
        -
        REFERENCE_SIZE // 2
    )

    reference_y = (
        target_center_y
        -
        REFERENCE_SIZE // 2
    )

    # --------------------------------------------------------
    # Add target-specific local structures.
    # --------------------------------------------------------

    (
        layout,
        target_features
    ) = add_subtle_target_features(
        layout,
        target_center_x,
        target_center_y
    )

    # --------------------------------------------------------
    # Extract reference from high-resolution physical layout.
    # --------------------------------------------------------

    reference = layout[
        reference_y:
        reference_y + REFERENCE_SIZE,

        reference_x:
        reference_x + REFERENCE_SIZE
    ].copy()

    # --------------------------------------------------------
    # Extract search field.
    # --------------------------------------------------------

    search_physical = layout[
        search_origin_y:
        search_origin_y + search_physical_size,

        search_origin_x:
        search_origin_x + search_physical_size
    ].copy()

    # --------------------------------------------------------
    # Downsample physical search field to exactly 1000x1000.
    # --------------------------------------------------------

    search = cv2.resize(
        search_physical,
        (
            SEARCH_SIZE,
            SEARCH_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Ground truth target center in search-image pixels.
    #
    # This is calculated from the actual physical geometry.
    # --------------------------------------------------------

    target_search_x = (
        target_center_x
        -
        search_origin_x
    ) / scale_factor

    target_search_y = (
        target_center_y
        -
        search_origin_y
    ) / scale_factor

    # --------------------------------------------------------
    # Independent capture transformations.
    # --------------------------------------------------------

    (
        reference_augmented,
        reference_rotation,
        reference_scale
    ) = augment_reference(
        reference
    )

    search_augmented = augment_search(
        search
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    sample_dir = os.path.join(
        output_dir,
        f"sample_{sample_id:04d}"
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    save_image(
        os.path.join(
            sample_dir,
            "reference.png"
        ),
        reference_augmented
    )

    save_image(
        os.path.join(
            sample_dir,
            "search.png"
        ),
        search_augmented
    )

    # --------------------------------------------------------
    # Ground-truth metadata.
    # --------------------------------------------------------

    metadata = {

        "architecture": "DRAM",

        "reference_image": {
            "width": REFERENCE_SIZE,
            "height": REFERENCE_SIZE
        },

        "search_image": {
            "width": SEARCH_SIZE,
            "height": SEARCH_SIZE
        },

        "scale_factor": float(
            scale_factor
        ),

        "reference_scale_variation": float(
            reference_scale
        ),

        "reference_rotation_deg": float(
            reference_rotation
        ),

        "reference_start_high_res": {
            "x": int(reference_x),
            "y": int(reference_y)
        },

        "search_crop_origin_high_res": {
            "x": int(search_origin_x),
            "y": int(search_origin_y)
        },

        "target_center_high_res": {
            "x": int(target_center_x),
            "y": int(target_center_y)
        },

        "target_center_search": {
            "x": float(target_search_x),
            "y": float(target_search_y)
        },

        "target_features": target_features,

        "generator_requirements": {

            "independent_reference_noise": True,

            "independent_search_noise": True,

            "edge_brightening": True,

            "blur": True,

            "rotation_variation": True,

            "scaling_variation": True,

            "search_has_higher_noise": True,

            "known_ground_truth": True,

            "periodic_dram_layout": True,

            "subtle_target_local_variations": True
        }
    }

    with open(
        os.path.join(
            sample_dir,
            "ground_truth.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "DRIFT-SENSE DRAM synthetic "
            "dataset generator"
        )
    )

    parser.add_argument(
        "--num_pairs",
        type=int,
        default=DEFAULT_NUM_PAIRS
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = parser.parse_args()

    if args.num_pairs < 30:

        raise ValueError(
            "The hackathon requires at least "
            "30 randomized image pairs."
        )

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Remove previous dataset.
    # --------------------------------------------------------

    if os.path.exists(
        args.output_dir
    ):

        print(
            f"Removing existing dataset: "
            f"{args.output_dir}"
        )

        shutil.rmtree(
            args.output_dir
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("DRIFT-SENSE DRAM DATASET GENERATOR")
    print("=" * 60)
    print(
        f"Pairs              : {args.num_pairs}"
    )
    print(
        f"Output             : {args.output_dir}"
    )
    print(
        f"Random seed        : {args.seed}"
    )
    print(
        f"Scale range        : "
        f"{MIN_SCALE:.1f}x - "
        f"{MAX_SCALE:.1f}x"
    )
    print(
        f"Reference          : "
        f"{REFERENCE_SIZE}x{REFERENCE_SIZE}"
    )
    print(
        f"Search             : "
        f"{SEARCH_SIZE}x{SEARCH_SIZE}"
    )
    print(
        "Architecture       : DRAM"
    )
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Generate samples.
    # --------------------------------------------------------

    for sample_id in range(
        args.num_pairs
    ):

        print(
            f"[{sample_id + 1:04d}/"
            f"{args.num_pairs:04d}] "
            f"Generating sample..."
        )

        generate_sample(
            sample_id,
            args.output_dir
        )

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(
        f"Generated {args.num_pairs} DRAM pairs."
    )
    print(
        f"Location: {args.output_dir}"
    )
    print()
    print(
        "Every sample contains:"
    )
    print(
        "  reference.png"
    )
    print(
        "  search.png"
    )
    print(
        "  ground_truth.json"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()