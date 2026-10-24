"""Texture descriptors (DCT, LBP, DWT, Gabor) shared by Week 3 tasks."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern


METHODS = ("dct", "lbp", "dwt", "gabor")
DISTANCE_NAMES = ("l1", "l2", "chi2", "hellinger")


@dataclass(frozen=True)
class TextureDescriptorConfig:
    """Configuration for a block-based grayscale texture descriptor.

    Common fields
    -------------
    method : {"dct", "lbp", "dwt", "gabor"}
    grid : number of blocks per side (grid x grid blocks).
    size : the image is resized to size x size before the descriptor (None = keep).
    denoise : apply the 3x3 median filter first (Week 3 Task 1 result).
    distance : {"l1", "l2", "chi2", "hellinger"} used to compare descriptors.

    Method-specific fields (ignored by the other methods)
    -----------------------------------------------------
    dct   : block_size, n_coefs, skip_dc
    lbp   : scales ((P, R), ...), uniform
    dwt   : wavelet, levels
    gabor : wavelengths, n_orient, sigma_factor
    """

    method: str = "lbp"
    grid: int = 16
    size: int | None = 256
    denoise: bool = True
    distance: str = "l1"
    # DCT
    block_size: int = 16
    n_coefs: int = 5
    skip_dc: bool = True
    # LBP
    scales: tuple[tuple[int, float], ...] = ((8, 1.0),)
    uniform: bool = False
    # DWT
    wavelet: str = "haar"
    levels: int = 2
    # Gabor
    wavelengths: tuple[float, ...] = (4, 8, 16)
    n_orient: int = 6
    sigma_factor: float = 0.56

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.grid <= 0:
            raise ValueError("grid must be positive")
        if self.size is not None and self.size <= 0:
            raise ValueError("size must be positive or None")
        if self.size is not None and self.size < self.grid:
            raise ValueError("size cannot be smaller than grid")
        if self.distance not in DISTANCE_NAMES:
            raise ValueError(f"distance must be one of {DISTANCE_NAMES}")

        if self.method == "dct":
            if self.block_size <= 0:
                raise ValueError("block_size must be positive")
            if self.n_coefs <= 0 or self.n_coefs + int(self.skip_dc) > self.block_size**2:
                raise ValueError("n_coefs does not fit in a block_size x block_size DCT")
        elif self.method == "lbp":
            if not self.scales or any(p <= 0 or r <= 0 for p, r in self.scales):
                raise ValueError("scales must contain positive (P, R) pairs")
        elif self.method == "dwt":
            if self.levels <= 0:
                raise ValueError("levels must be positive")
            pywt.Wavelet(self.wavelet)  # raises ValueError for unknown wavelets
        else:  # gabor
            if not self.wavelengths or any(w <= 0 for w in self.wavelengths):
                raise ValueError("wavelengths must contain positive values")
            if self.n_orient <= 0:
                raise ValueError("n_orient must be positive")
            if self.sigma_factor <= 0:
                raise ValueError("sigma_factor must be positive")


# Selected configurations (QSD1-W3 sweeps, Task 2). All use L1.
WEEK3_DCT = TextureDescriptorConfig(method="dct", grid=8, block_size=16, n_coefs=5)
WEEK3_LBP = TextureDescriptorConfig(method="lbp", grid=16)
WEEK3_DWT_HAAR = TextureDescriptorConfig(method="dwt", grid=12, wavelet="haar", levels=2)
WEEK3_DWT_DB2 = TextureDescriptorConfig(method="dwt", grid=8, wavelet="db2", levels=3)
WEEK3_GABOR = TextureDescriptorConfig(
    method="gabor", grid=4, wavelengths=(3, 6, 12), n_orient=4
)

# LBP variants tuned on QSD2-W3 (Task 4). The last one is the final method.
WEEK3_LBP_R2_G12_L1 = TextureDescriptorConfig(
    method="lbp", grid=12, scales=((8, 2.0),), distance="l1"
)
WEEK3_LBP_R2_G16_HELLINGER = TextureDescriptorConfig(
    method="lbp", grid=16, scales=((8, 2.0),), distance="hellinger"
)

WEEK3_FINAL_CONFIG = WEEK3_LBP_R2_G16_HELLINGER

TEXTURE_PRESETS = {
    "DCT (8x8 blocks, 5 coefficients, L1)": WEEK3_DCT,
    "LBP (8 neighbours, radius 1, 16x16 blocks, L1)": WEEK3_LBP,
    "DWT haar, 2 levels, grid 12, L1": WEEK3_DWT_HAAR,
    "DWT db2, 3 levels, grid 8, L1": WEEK3_DWT_DB2,
    "Gabor (3, 6, 12), 4 orientations, grid 4, L1": WEEK3_GABOR,
    "LBP radius 2, grid 12, L1": WEEK3_LBP_R2_G12_L1,
    "LBP radius 2, grid 16, Hellinger": WEEK3_LBP_R2_G16_HELLINGER,
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def preprocess(img_bgr: np.ndarray, denoise: bool = True) -> np.ndarray:
    return cv2.medianBlur(img_bgr, 3) if denoise else img_bgr


def texture_input(
    image: np.ndarray, size: int | None = 256, denoise: bool = True
) -> np.ndarray:
    """Common pipeline for queries and database: median 3x3 + resize + gray."""
    image = preprocess(image, denoise)
    if size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return to_gray(image)


def _grid_edges(length: int, grid: int) -> np.ndarray:
    return np.linspace(0, length, grid + 1, dtype=int)


def _l2_normalise(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------
def zigzag_indices(n: int) -> list[tuple[int, int]]:
    idx = [(i, j) for i in range(n) for j in range(n)]
    return sorted(idx, key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))


_ZZ: dict[int, list[tuple[int, int]]] = {}


def dct_descriptor(
    gray: np.ndarray,
    grid: int = 8,
    block_size: int = 16,
    n_coefs: int = 5,
    skip_dc: bool = True,
) -> np.ndarray:
    """Zig-zag low-frequency DCT coefficients of every block, L2-normalised."""
    if block_size not in _ZZ:
        _ZZ[block_size] = zigzag_indices(block_size)
    start = 1 if skip_dc else 0
    zz = _ZZ[block_size][start : start + n_coefs]
    ys = _grid_edges(gray.shape[0], grid)
    xs = _grid_edges(gray.shape[1], grid)
    feats = []
    for i in range(grid):
        for j in range(grid):
            blk = gray[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            blk = cv2.resize(
                blk, (block_size, block_size), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            c = cv2.dct(blk)
            feats.append([c[a, b] for a, b in zz])
    return _l2_normalise(np.asarray(feats, np.float32).ravel())


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------
def block_hist(code_img: np.ndarray, grid: int, n_bins: int) -> np.ndarray:
    """Normalised histogram of the codes of every block, concatenated."""
    ys = _grid_edges(code_img.shape[0], grid)
    xs = _grid_edges(code_img.shape[1], grid)
    out = []
    for i in range(grid):
        for j in range(grid):
            blk = code_img[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            hist = np.bincount(blk.ravel(), minlength=n_bins).astype(np.float32)
            out.append(hist / max(hist.sum(), 1e-12))
    return np.concatenate(out) / grid**2


def lbp_descriptor(
    gray: np.ndarray,
    grid: int = 4,
    scales: Sequence[tuple[int, float]] = ((8, 1.0),),
    uniform: bool = False,
) -> np.ndarray:
    """Block LBP histograms. P=8 without `uniform` gives the 256-code LBP at any
    radius; every other case uses the uniform pattern (P + 2 bins)."""
    parts = []
    for P, R in scales:
        method = "default" if (P == 8 and not uniform) else "uniform"
        codes = local_binary_pattern(gray, P, R, method=method).astype(np.int32)
        n_bins = 256 if method == "default" else P + 2
        parts.append(block_hist(codes, grid, n_bins))
    d = np.concatenate(parts)
    return d / (d.sum() + 1e-12)


# ---------------------------------------------------------------------------
# DWT
# ---------------------------------------------------------------------------
def _dwt_level(block_side: int, wavelet: str, levels: int) -> int:
    """Largest number of levels <= `levels` that a block of this size supports."""
    return min(levels, pywt.dwt_max_level(block_side, pywt.Wavelet(wavelet).dec_len))


def dwt_descriptor(
    gray: np.ndarray, grid: int = 8, wavelet: str = "haar", levels: int = 2
) -> np.ndarray:
    """Mean and std of |detail coefficients| per block, band and level, L2-normalised."""
    img = gray.astype(np.float32) / 255.0
    ys = _grid_edges(img.shape[0], grid)
    xs = _grid_edges(img.shape[1], grid)
    level = _dwt_level(int(min(np.diff(ys).min(), np.diff(xs).min())), wavelet, levels)
    feats = []
    for i in range(grid):
        for j in range(grid):
            blk = img[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            coeffs = pywt.wavedec2(blk, wavelet, level=level)  # [approx, (H, V, D) level L, ..., level 1]
            for detail in coeffs[1:]:
                for band in detail:
                    mag = np.abs(band)
                    feats.extend((mag.mean(), mag.std()))
    return _l2_normalise(np.asarray(feats, np.float32))


# ---------------------------------------------------------------------------
# Gabor
# ---------------------------------------------------------------------------
_GABOR: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}


def gabor_bank(
    wavelengths: Sequence[float] = (4, 8, 16),
    n_orient: int = 6,
    sigma_factor: float = 0.56,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (even, odd) kernels. sigma = 0.56 * wavelength gives roughly one octave of bandwidth."""
    key = (tuple(wavelengths), n_orient, sigma_factor)
    if key not in _GABOR:
        bank = []
        for lam in wavelengths:
            sigma = sigma_factor * lam
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            for k in range(n_orient):
                theta = np.pi * k / n_orient
                even = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, 1.0, 0.0, ktype=cv2.CV_32F
                )
                odd = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, 1.0, np.pi / 2, ktype=cv2.CV_32F
                )
                bank.append((even - even.mean(), odd))
        _GABOR[key] = bank
    return _GABOR[key]


def gabor_responses(
    gray: np.ndarray,
    wavelengths: Sequence[float] = (4, 8, 16),
    n_orient: int = 6,
    sigma_factor: float = 0.56,
) -> list[np.ndarray]:
    """Magnitude of the complex Gabor response for every filter of the bank."""
    img = gray.astype(np.float32) / 255.0
    out = []
    for even, odd in gabor_bank(wavelengths, n_orient, sigma_factor):
        re = cv2.filter2D(img, cv2.CV_32F, even, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(img, cv2.CV_32F, odd, borderType=cv2.BORDER_REFLECT)
        out.append(np.sqrt(re**2 + im**2))
    return out


def gabor_descriptor(
    gray: np.ndarray,
    grid: int = 8,
    wavelengths: Sequence[float] = (4, 8, 16),
    n_orient: int = 6,
    sigma_factor: float = 0.56,
) -> np.ndarray:
    """Mean and std of the response magnitude per filter and block, L2-normalised."""
    ys = _grid_edges(gray.shape[0], grid)
    xs = _grid_edges(gray.shape[1], grid)
    feats = []
    for mag in gabor_responses(gray, wavelengths, n_orient, sigma_factor):
        for i in range(grid):
            for j in range(grid):
                blk = mag[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
                feats.extend((blk.mean(), blk.std()))
    return _l2_normalise(np.asarray(feats, np.float32))


# ---------------------------------------------------------------------------
# Configured descriptors
# ---------------------------------------------------------------------------
def descriptor(gray: np.ndarray, config: TextureDescriptorConfig) -> np.ndarray:
    """Descriptor of an already preprocessed grayscale image."""
    if config.method == "dct":
        return dct_descriptor(
            gray, config.grid, config.block_size, config.n_coefs, config.skip_dc
        )
    if config.method == "lbp":
        return lbp_descriptor(gray, config.grid, config.scales, config.uniform)
    if config.method == "dwt":
        return dwt_descriptor(gray, config.grid, config.wavelet, config.levels)
    return gabor_descriptor(
        gray, config.grid, config.wavelengths, config.n_orient, config.sigma_factor
    )


def describe_images(
    images: Sequence[np.ndarray], config: TextureDescriptorConfig = WEEK3_FINAL_CONFIG
) -> np.ndarray:
    """Describe an ordered collection of BGR images (median + resize + gray + descriptor)."""
    return np.asarray(
        [
            descriptor(texture_input(image, config.size, config.denoise), config)
            for image in images
        ],
        dtype=np.float32,
    )


def descriptor_length(config: TextureDescriptorConfig) -> int:
    """Number of values per image, measured on a blank image of the working size."""
    side = config.size or 256
    return len(descriptor(np.zeros((side, side), np.uint8), config))


# ---------------------------------------------------------------------------
# Distances and fusion
# ---------------------------------------------------------------------------
def hellinger_dist(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.sqrt(((np.sqrt(q) - np.sqrt(D)) ** 2).sum(axis=1))


def chi2(q: np.ndarray, D: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return 0.5 * (((q - D) ** 2) / (q + D + eps)).sum(axis=1)


def l1(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.abs(q - D).sum(axis=1)


def l2(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.sqrt(((q - D) ** 2).sum(axis=1))


DISTANCES = {"l1": l1, "l2": l2, "chi2": chi2, "hellinger": hellinger_dist}


def compute_distances(
    query_features: np.ndarray, database_features: np.ndarray, distance: str = "l1"
) -> np.ndarray:
    """(queries x database) distance matrix."""
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCE_NAMES}")
    function = DISTANCES[distance]
    return np.vstack([function(query, database_features) for query in query_features])