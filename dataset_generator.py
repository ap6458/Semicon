import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ============================================================
# CONFIGURATION
# ============================================================

HIGH_RES_SIZE = 10000       # 10,000 x 10,000 nm
REFERENCE_SIZE = 1000       # 1,000 x 1,000 nm
SEARCH_SIZE = 1000          # 1,000 x 1,000 pixels
SCALE_FACTOR = 10

NUM_SAMPLES = 50            # Start with 50, then increase to 500/1000

OUTPUT_DIR = "dataset"

# ============================================================
# RANDOM SEED
# ============================================================

# Keep None for genuinely different datasets every run.
# Set to an integer if you want reproducibility.
SEED = None

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)


# ============================================================
# UTILITY
# ============================================================

def random_choice(options):
    return random.choice(options)


def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# DRAM STRUCTURE GENERATOR
# ============================================================

def generate_dram_layout(size=HIGH_RES_SIZE):
    """
    Generate a synthetic DRAM-style semiconductor layout.

    Main structures:
        - Horizontal word lines
        - Vertical bit lines
        - Contacts / vias at intersections
        - Local structural variation
        - Small defects / irregularities
    """

    img = np.zeros((size, size), dtype=np.float32)

    # --------------------------------------------------------
    # GLOBAL STRUCTURAL PARAMETERS
    # --------------------------------------------------------

    wordline_spacing = random.randint(75, 135)
    bitline_spacing = random.randint(75, 135)

    wordline_width = random.randint(5, 16)
    bitline_width = random.randint(5, 16)

    background = random.randint(30, 75)

    wordline_intensity = random.randint(130, 205)
    bitline_intensity = random.randint(130, 205)

    img[:] = background

    # --------------------------------------------------------
    # WORD LINES
    # --------------------------------------------------------

    wordline_positions = []

    start_y = random.randint(
        wordline_spacing // 3,
        wordline_spacing
    )

    y = start_y

    while y < size:

        # Small local positional variation
        jitter = random.randint(-2, 2)

        yy = y + jitter

        width = max(
            2,
            wordline_width + random.randint(-2, 2)
        )

        intensity = np.clip(
            wordline_intensity + random.randint(-15, 15),
            80,
            240
        )

        y1 = max(0, yy - width // 2)
        y2 = min(size, yy + width // 2 + 1)

        img[y1:y2, :] = intensity

        wordline_positions.append(yy)

        y += wordline_spacing + random.randint(-4, 4)

    # --------------------------------------------------------
    # BIT LINES
    # --------------------------------------------------------

    bitline_positions = []

    start_x = random.randint(
        bitline_spacing // 3,
        bitline_spacing
    )

    x = start_x

    while x < size:

        jitter = random.randint(-2, 2)

        xx = x + jitter

        width = max(
            2,
            bitline_width + random.randint(-2, 2)
        )

        intensity = np.clip(
            bitline_intensity + random.randint(-15, 15),
            80,
            240
        )

        x1 = max(0, xx - width // 2)
        x2 = min(size, xx + width // 2 + 1)

        img[:, x1:x2] = intensity

        bitline_positions.append(xx)

        x += bitline_spacing + random.randint(-4, 4)

    # --------------------------------------------------------
    # CONTACT / VIA STRUCTURES
    # --------------------------------------------------------

    contact_style = random_choice([
        "circle",
        "square",
        "mixed"
    ])

    base_contact_radius = random.randint(4, 10)

    for y in wordline_positions:

        for x in bitline_positions:

            radius = max(
                2,
                base_contact_radius + random.randint(-2, 3)
            )

            intensity = random.randint(205, 255)

            y1 = max(0, y - radius)
            y2 = min(size, y + radius + 1)

            x1 = max(0, x - radius)
            x2 = min(size, x + radius + 1)

            yy, xx = np.ogrid[
                y1:y2,
                x1:x2
            ]

            if contact_style == "circle":

                mask = (
                    (yy - y) ** 2 +
                    (xx - x) ** 2
                    <= radius ** 2
                )

            elif contact_style == "square":

                mask = np.ones(
                    (y2 - y1, x2 - x1),
                    dtype=bool
                )

            else:

                if random.random() < 0.5:

                    mask = (
                        (yy - y) ** 2 +
                        (xx - x) ** 2
                        <= radius ** 2
                    )

                else:

                    mask = np.ones(
                        (y2 - y1, x2 - x1),
                        dtype=bool
                    )

            region = img[y1:y2, x1:x2]

            region[mask] = intensity

    # --------------------------------------------------------
    # LOCAL STRUCTURAL FEATURES
    # --------------------------------------------------------

    # Add occasional small blocks/regions.
    # These make certain locations more distinctive.
    num_features = random.randint(3, 12)

    for _ in range(num_features):

        fx = random.randint(0, size - 100)
        fy = random.randint(0, size - 100)

        fw = random.randint(20, 100)
        fh = random.randint(20, 100)

        feature_intensity = random.randint(
            70,
            180
        )

        img[
            fy:fy + fh,
            fx:fx + fw
        ] = feature_intensity

    # --------------------------------------------------------
    # OCCASIONAL MISSING CONTACTS
    # --------------------------------------------------------

    # Small number of missing contacts.
    # This creates realistic structural diversity.
    defect_probability = random.uniform(
        0.001,
        0.01
    )

    for y in wordline_positions:

        for x in bitline_positions:

            if random.random() < defect_probability:

                radius = base_contact_radius + 2

                y1 = max(0, y - radius)
                y2 = min(size, y + radius + 1)

                x1 = max(0, x - radius)
                x2 = min(size, x + radius + 1)

                img[
                    y1:y2,
                    x1:x2
                ] *= random.uniform(
                    0.2,
                    0.6
                )

    # --------------------------------------------------------
    # GLOBAL CONTRAST VARIATION
    # --------------------------------------------------------

    contrast = random.uniform(
        0.85,
        1.20
    )

    mean = np.mean(img)

    img = (
        (img - mean) * contrast
        + mean
    )

    return np.clip(img, 0, 255)


# ============================================================
# EDGE BRIGHTENING
# ============================================================

def apply_edge_brightening(img, strength=None):

    if strength is None:
        strength = random.uniform(
            0.25,
            0.75
        )

    pil_img = Image.fromarray(
        clamp_uint8(img)
    )

    blur_radius = random.uniform(
        1.0,
        3.0
    )

    blurred = pil_img.filter(
        ImageFilter.GaussianBlur(
            radius=blur_radius
        )
    )

    blur_array = np.asarray(
        blurred
    ).astype(np.float32)

    edges = img - blur_array

    result = img + strength * edges

    return np.clip(
        result,
        0,
        255
    )


# ============================================================
# NOISE
# ============================================================

def add_gaussian_noise(
    img,
    sigma_min,
    sigma_max
):

    sigma = random.uniform(
        sigma_min,
        sigma_max
    )

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    )

    return np.clip(
        img + noise,
        0,
        255
    )


def add_poisson_noise(img):

    # Normalize to approximately 0-1
    normalized = np.clip(
        img / 255.0,
        0,
        1
    )

    # Random photon-count scale
    scale = random.choice([
        30,
        50,
        80,
        120
    ])

    noisy = np.random.poisson(
        normalized * scale
    ) / scale

    return np.clip(
        noisy * 255,
        0,
        255
    )


# ============================================================
# BLUR
# ============================================================

def apply_blur(img):

    if random.random() < 0.65:

        radius = random.uniform(
            0.2,
            1.2
        )

        pil_img = Image.fromarray(
            clamp_uint8(img)
        )

        pil_img = pil_img.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )

        return np.asarray(
            pil_img
        ).astype(np.float32)

    return img


# ============================================================
# BRIGHTNESS / CONTRAST
# ============================================================

def apply_capture_variation(img):

    pil_img = Image.fromarray(
        clamp_uint8(img)
    )

    brightness_factor = random.uniform(
        0.85,
        1.15
    )

    contrast_factor = random.uniform(
        0.85,
        1.20
    )

    pil_img = ImageEnhance.Brightness(
        pil_img
    ).enhance(
        brightness_factor
    )

    pil_img = ImageEnhance.Contrast(
        pil_img
    ).enhance(
        contrast_factor
    )

    return np.asarray(
        pil_img
    ).astype(np.float32)


# ============================================================
# ROTATION
# ============================================================

def rotate_image(
    img,
    max_angle=3.0
):

    angle = random.uniform(
        -max_angle,
        max_angle
    )

    pil_img = Image.fromarray(
        clamp_uint8(img)
    )

    rotated = pil_img.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=int(np.mean(img))
    )

    return (
        np.asarray(rotated)
        .astype(np.float32),
        angle
    )


# ============================================================
# REFERENCE AUGMENTATION
# ============================================================

def augment_reference(reference):

    # Independent blur
    reference = apply_blur(
        reference
    )

    # Independent edge brightening
    reference = apply_edge_brightening(
        reference,
        strength=random.uniform(
            0.25,
            0.65
        )
    )

    # Independent brightness/contrast
    reference = apply_capture_variation(
        reference
    )

    # Reference capture can have a small orientation difference
    reference, rotation_angle = rotate_image(
        reference,
        max_angle=2.0
    )

    # Moderate noise
    reference = add_gaussian_noise(
        reference,
        2.0,
        7.0
    )

    # Occasionally add Poisson-like noise
    if random.random() < 0.35:

        reference = add_poisson_noise(
            reference
        )

    return (
        np.clip(
            reference,
            0,
            255
        ),
        rotation_angle
    )


# ============================================================
# SEARCH IMAGE AUGMENTATION
# ============================================================

def augment_search(search):

    # Search is intentionally more degraded.
    search = apply_blur(
        search
    )

    # Stronger edge enhancement
    search = apply_edge_brightening(
        search,
        strength=random.uniform(
            0.35,
            0.80
        )
    )

    # Brightness / contrast variation
    search = apply_capture_variation(
        search
    )

    # Stronger Gaussian noise
    search = add_gaussian_noise(
        search,
        7.0,
        20.0
    )

    # Occasionally add Poisson-like noise
    if random.random() < 0.55:

        search = add_poisson_noise(
            search
        )

    return np.clip(
        search,
        0,
        255
    )


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_sample(sample_id):

    print(
        f"Generating sample "
        f"{sample_id + 1}/{NUM_SAMPLES}"
    )

    # --------------------------------------------------------
    # 1. Generate a completely new DRAM structure
    # --------------------------------------------------------

    large_layout = generate_dram_layout(
        HIGH_RES_SIZE
    )

    # --------------------------------------------------------
    # 2. Random target location
    # --------------------------------------------------------

    max_start = (
        HIGH_RES_SIZE -
        REFERENCE_SIZE
    )

    ref_x = random.randint(
        0,
        max_start
    )

    ref_y = random.randint(
        0,
        max_start
    )

    # --------------------------------------------------------
    # 3. Extract high-resolution reference
    # --------------------------------------------------------

    reference = large_layout[
        ref_y:
        ref_y + REFERENCE_SIZE,

        ref_x:
        ref_x + REFERENCE_SIZE
    ].copy()

    # --------------------------------------------------------
    # 4. Downsample entire 10,000 x 10,000 structure
    #    to 1,000 x 1,000.
    #
    #    THIS preserves the exact 10× scale relationship.
    # --------------------------------------------------------

    large_pil = Image.fromarray(
        clamp_uint8(
            large_layout
        )
    )

    search_pil = large_pil.resize(
        (
            SEARCH_SIZE,
            SEARCH_SIZE
        ),
        Image.Resampling.LANCZOS
    )

    search = np.asarray(
        search_pil
    ).astype(np.float32)

    # --------------------------------------------------------
    # 5. Apply independent capture conditions
    # --------------------------------------------------------

    reference, reference_rotation = (
        augment_reference(
            reference
        )
    )

    search = augment_search(
        search
    )

    # --------------------------------------------------------
    # 6. Ground truth
    #
    # The physical reference region is:
    #
    # ref_x ... ref_x + 1000
    # ref_y ... ref_y + 1000
    #
    # After 10× downsampling:
    # target center is divided by 10.
    # --------------------------------------------------------

    target_x = (
        ref_x +
        REFERENCE_SIZE / 2
    ) / SCALE_FACTOR

    target_y = (
        ref_y +
        REFERENCE_SIZE / 2
    ) / SCALE_FACTOR

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_id:04d}"
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    Image.fromarray(
        clamp_uint8(reference)
    ).save(
        os.path.join(
            sample_dir,
            "reference.png"
        )
    )

    Image.fromarray(
        clamp_uint8(search)
    ).save(
        os.path.join(
            sample_dir,
            "search.png"
        )
    )

    # --------------------------------------------------------
    # 8. Ground-truth metadata
    # --------------------------------------------------------

    metadata = {

        "architecture": "DRAM",

        "image_size": [
            1000,
            1000
        ],

        "scale_factor": 10,

        "reference_start_high_res": {
            "x": ref_x,
            "y": ref_y
        },

        "reference_center_high_res": {
            "x": ref_x + REFERENCE_SIZE / 2,
            "y": ref_y + REFERENCE_SIZE / 2
        },

        "target_center_search": {
            "x": target_x,
            "y": target_y
        },

        "reference_rotation_deg": (
            reference_rotation
        ),

        "generator": {
            "wordline_spacing": "randomized",
            "bitline_spacing": "randomized",
            "line_width": "randomized",
            "contact_size": "randomized",
            "contact_shape": "circle/square/mixed",
            "local_features": True,
            "missing_contacts": True,
            "edge_brightening": True,
            "independent_noise": True,
            "search_more_noisy": True,
            "blur": True,
            "brightness_variation": True,
            "contrast_variation": True,
            "rotation_variation": True
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

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for i in range(NUM_SAMPLES):

        generate_sample(i)

    print(
        "\n===================================="
    )

    print(
        "Dataset generation complete!"
    )

    print(
        f"Generated {NUM_SAMPLES} image pairs."
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        "===================================="
    )