from __future__ import annotations

from _venv_bootstrap import rerun_with_nearest_venv

rerun_with_nearest_venv()

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply basic image filters and Sobel edge detection to a local image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the local image file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated result images.",
    )
    return parser.parse_args()


def load_grayscale_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    # Use edge padding so the output keeps the same size without adding dark borders.
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row : row + kernel_height, col : col + kernel_width]
            output[row, col] = float(np.sum(region * kernel))

    return output


def median_filter(image: np.ndarray, window_size: int = 3) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row : row + window_size, col : col + window_size]
            output[row, col] = float(np.median(region))

    return output


def sobel_filter(image: np.ndarray) -> np.ndarray:
    sobel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )
    sobel_y = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ],
        dtype=np.float32,
    )

    grad_x = convolve2d(image, sobel_x)
    grad_y = convolve2d(image, sobel_y)
    magnitude = np.hypot(grad_x, grad_y)

    max_value = float(magnitude.max())
    if max_value > 0:
        magnitude = magnitude / max_value * 255.0

    return magnitude


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(output_path)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input image does not exist: {args.input}")

    image = load_grayscale_image(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    average_kernel = np.full((3, 3), 1 / 9, dtype=np.float32)
    weighted_average_kernel = np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=np.float32,
    ) / 16.0

    results = {
        "average": convolve2d(image, average_kernel),
        "weighted-average": convolve2d(image, weighted_average_kernel),
        "median": median_filter(image),
        "sobel": sobel_filter(image),
    }

    input_stem = args.input.stem
    print(f"Loaded image: {args.input.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    for name, result in results.items():
        output_path = output_dir / f"{input_stem}-{name}.png"
        save_image(result, output_path)
        print(f"Saved {name}: {output_path.resolve()}")


if __name__ == "__main__":
    main()
