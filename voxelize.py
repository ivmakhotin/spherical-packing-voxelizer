#!/usr/bin/env python3
"""
Voxelizer for sphere packing data (DEM output) → LBPM binary geometry.

CSV format expected: ID, R, X, Y, Z

Output: raw binary file (int8), shape (Nz, Ny, Nx), C-order
  0  = solid (inside a sphere)
  1  = pore/fluid space

Usage:
  python voxelize.py ball_data.csv [options]

The companion .db config file is auto-generated alongside the .raw file.
"""

import argparse
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── optional numba acceleration ───────────────────────────────────────────────
try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _coverage_kernel_nb(
        sx_arr, sy_arr, sz_arr, sr_arr,
        k_min_arr, k_max_arr,
        ix0_arr, ix1_arr, iy0_arr, iy1_arr,
        scx, scy, scz,
        sub_spacing, SNx, SNy, n_sub, Nz, Ny, Nx,
        zmin, min_count, solid_label, pore_label,
    ):
        """Numba parallel coverage kernel: one thread per Z-slice via prange."""
        grid = np.empty((Nz, Ny, Nx), dtype=np.int8)
        for k in numba.prange(Nz):
            sub_slice = np.zeros((n_sub, SNy, SNx), dtype=np.bool_)
            sz0_sub = k * n_sub
            for si in range(len(sx_arr)):
                if k_min_arr[si] > k or k_max_arr[si] < k:
                    continue
                sx = sx_arr[si]
                sy = sy_arr[si]
                sz = sz_arr[si]
                sr = sr_arr[si]
                sr2 = sr * sr
                local_iz0 = max(0, int((sz - sr - zmin) / sub_spacing) - sz0_sub)
                local_iz1 = min(n_sub - 1, int((sz + sr - zmin) / sub_spacing) - sz0_sub)
                if local_iz0 > local_iz1:
                    continue
                ix0 = ix0_arr[si]
                ix1 = ix1_arr[si]
                iy0 = iy0_arr[si]
                iy1 = iy1_arr[si]
                if ix0 > ix1 or iy0 > iy1:
                    continue
                for dz in range(local_iz0, local_iz1 + 1):
                    z_d2 = (scz[sz0_sub + dz] - sz) ** 2
                    for dy in range(iy0, iy1 + 1):
                        yz_d2 = z_d2 + (scy[dy] - sy) ** 2
                        for dx in range(ix0, ix1 + 1):
                            if yz_d2 + (scx[dx] - sx) ** 2 <= sr2:
                                sub_slice[dz, dy, dx] = True
            # Downsample: count solid sub-voxels per parent voxel
            for j in range(Ny):
                for i in range(Nx):
                    cnt = 0
                    for dz in range(n_sub):
                        for dy in range(n_sub):
                            for dx in range(n_sub):
                                if sub_slice[dz, j * n_sub + dy, i * n_sub + dx]:
                                    cnt += 1
                    grid[k, j, i] = solid_label if cnt >= min_count else pore_label
        return grid

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Voxelize sphere packing CSV for LBPM simulation."
    )
    p.add_argument("csv", help="Input CSV file (columns: ID, R, X, Y, Z)")

    # Domain bounds – default: auto from data + 1 radius margin
    p.add_argument("--xmin", type=float, default=None, help="X lower bound")
    p.add_argument("--xmax", type=float, default=None, help="X upper bound")
    p.add_argument("--ymin", type=float, default=None, help="Y lower bound")
    p.add_argument("--ymax", type=float, default=None, help="Y upper bound")
    p.add_argument("--zmin", type=float, default=None, help="Z lower bound")
    p.add_argument("--zmax", type=float, default=None, help="Z upper bound")

    p.add_argument(
        "--spacing", "-s", type=float, default=0.1,
        help="Voxel size (same units as coordinates). Default: 0.1"
    )
    p.add_argument(
        "--voxel-length", type=float, default=None, metavar="UM",
        help="Physical voxel size written to the .db file, in micrometres (µm). "
             "Defaults to --spacing if not set. Use this when your coordinates are "
             "not in µm (e.g. if coords are in metres, pass --voxel-length <spacing*1e6>)."
    )
    p.add_argument(
        "--output", "-o", default="geometry",
        help="Output base name (produces <name>.raw and <name>.db). Default: geometry"
    )
    p.add_argument(
        "--solid-label", type=int, default=0,
        help="Label value written for solid voxels. Default: 0"
    )
    p.add_argument(
        "--pore-label", type=int, default=1,
        help="Label value written for pore/fluid voxels. Default: 1"
    )

    # Voxelization mode
    p.add_argument(
        "--mode", choices=["center", "coverage"], default="center",
        help="Voxelization mode. 'center': solid if voxel centre is inside a sphere "
             "(fast). 'coverage': solid if sphere(s) cover >= threshold of voxel volume "
             "(more accurate at boundaries). Default: center"
    )
    p.add_argument(
        "--sub-samples", type=int, default=4, metavar="N",
        help="Sub-divisions per voxel axis for coverage mode (N^3 samples per voxel). "
             "Default: 4 (64 samples). Higher = more accurate but more memory/time."
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, metavar="F",
        help="Volume fraction threshold for coverage mode [0..1]. Default: 0.5"
    )

    # Cross-section slices (middle slice of each axis by default)
    p.add_argument(
        "--slice-x", type=int, nargs="*", metavar="I",
        help="Voxel indices along X to export as YZ cross-section PNGs. "
             "Defaults to middle slice. Pass specific indices to override."
    )
    p.add_argument(
        "--slice-y", type=int, nargs="*", metavar="J",
        help="Voxel indices along Y to export as XZ cross-section PNGs. "
             "Defaults to middle slice. Pass specific indices to override."
    )
    p.add_argument(
        "--slice-z", type=int, nargs="*", metavar="K",
        help="Voxel indices along Z to export as XY cross-section PNGs. "
             "Defaults to middle slice. Pass specific indices to override."
    )
    p.add_argument(
        "--no-slices", action="store_true",
        help="Disable PNG slice output entirely."
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Use voxelize_coverage_fast (pre-filtering + numba/multiprocessing) "
             "instead of the baseline coverage implementation. "
             "Only applies to --mode coverage."
    )

    return p.parse_args()


def auto_bounds(df, col, margin_col="R"):
    """Tight bounds padded by one max-radius."""
    margin = df[margin_col].max()
    return df[col].min() - margin, df[col].max() + margin


def voxelize(df, xmin, xmax, ymin, ymax, zmin, zmax, spacing,
             solid_label=0, pore_label=1):
    """
    Returns a (Nz, Ny, Nx) int8 array.
    Voxel centres are at xmin + (i+0.5)*spacing, etc.
    """
    # Voxel counts
    Nx = max(1, int(np.ceil((xmax - xmin) / spacing)))
    Ny = max(1, int(np.ceil((ymax - ymin) / spacing)))
    Nz = max(1, int(np.ceil((zmax - zmin) / spacing)))

    print(f"Grid: Nx={Nx}  Ny={Ny}  Nz={Nz}  ({Nx*Ny*Nz:,} voxels)")

    # Voxel centre coordinates
    cx = xmin + (np.arange(Nx) + 0.5) * spacing  # shape (Nx,)
    cy = ymin + (np.arange(Ny) + 0.5) * spacing  # shape (Ny,)
    cz = zmin + (np.arange(Nz) + 0.5) * spacing  # shape (Nz,)

    # Start with all pore, then paint solid
    grid = np.full((Nz, Ny, Nx), pore_label, dtype=np.int8)

    # For each sphere, find the bounding box of voxel indices it can affect
    for _, row in df.iterrows():
        sx, sy, sz, sr = row["X"], row["Y"], row["Z"], row["R"]

        # Index range to search (clipped to grid)
        ix0 = max(0, int((sx - sr - xmin) / spacing))
        ix1 = min(Nx - 1, int((sx + sr - xmin) / spacing))
        iy0 = max(0, int((sy - sr - ymin) / spacing))
        iy1 = min(Ny - 1, int((sy + sr - ymin) / spacing))
        iz0 = max(0, int((sz - sr - zmin) / spacing))
        iz1 = min(Nz - 1, int((sz + sr - zmin) / spacing))

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue  # sphere entirely outside domain

        # Sub-arrays of coordinates in the bounding box
        lx = cx[ix0:ix1+1]          # (nx,)
        ly = cy[iy0:iy1+1]          # (ny,)
        lz = cz[iz0:iz1+1]          # (nz,)

        # Squared distance from sphere centre – broadcast to (nz, ny, nx)
        dist2 = (
            (lz[:, None, None] - sz) ** 2 +
            (ly[None, :, None] - sy) ** 2 +
            (lx[None, None, :] - sx) ** 2
        )

        inside = dist2 <= sr ** 2
        grid[iz0:iz1+1, iy0:iy1+1, ix0:ix1+1][inside] = solid_label

    return grid, Nx, Ny, Nz


def voxelize_coverage(df, xmin, xmax, ymin, ymax, zmin, zmax, spacing,
                      solid_label=0, pore_label=1, n_sub=4, threshold=0.5):
    """
    Returns a (Nz, Ny, Nx) int8 array using sub-voxel volume sampling.

    Processes one Z-slice at a time so memory is O(n_sub * Ny*n_sub * Nx*n_sub)
    instead of O(Nz * Ny * Nx * n_sub^3).

    Each voxel is subdivided into n_sub^3 sub-voxels. A sub-voxel is solid if
    its centre is inside any sphere. The parent voxel is solid if the fraction
    of solid sub-voxels >= threshold.
    """
    Nx = max(1, int(np.ceil((xmax - xmin) / spacing)))
    Ny = max(1, int(np.ceil((ymax - ymin) / spacing)))
    Nz = max(1, int(np.ceil((zmax - zmin) / spacing)))

    print(f"Grid: Nx={Nx}  Ny={Ny}  Nz={Nz}  ({Nx*Ny*Nz:,} voxels)")

    sub_spacing = spacing / n_sub
    SNx, SNy = Nx * n_sub, Ny * n_sub
    SNz = Nz * n_sub

    slice_mb = n_sub * SNy * SNx / 1e6
    print(f"Coverage mode: {n_sub}^3={n_sub**3} sub-samples/voxel, threshold={threshold:.0%}")
    print(f"Peak memory per slice: ~{slice_mb:.1f} MB boolean  (was ~{slice_mb*Nz:.0f} MB total)")

    # Sub-voxel centre coordinates (full XY, full Z for index lookups)
    scx = xmin + (np.arange(SNx) + 0.5) * sub_spacing
    scy = ymin + (np.arange(SNy) + 0.5) * sub_spacing
    scz = zmin + (np.arange(SNz) + 0.5) * sub_spacing

    min_count = int(np.ceil(threshold * n_sub ** 3))
    grid = np.empty((Nz, Ny, Nx), dtype=np.int8)

    for k in range(Nz):
        if k % max(1, Nz // 10) == 0:
            print(f"  Z-slice {k}/{Nz} ({100*k//Nz}%)")

        # Sub-voxel z-index range for this voxel row
        sz0_sub = k * n_sub
        sz1_sub = (k + 1) * n_sub
        lz = scz[sz0_sub:sz1_sub]   # shape (n_sub,)

        # One Z-slice of the sub-grid: shape (n_sub, SNy, SNx)
        sub_slice = np.zeros((n_sub, SNy, SNx), dtype=bool)

        for _, row in df.iterrows():
            sx, sy, sz, sr = row["X"], row["Y"], row["Z"], row["R"]

            # Sub-voxel z-range of this sphere, clipped to current slice
            local_iz0 = max(0,        int((sz - sr - zmin) / sub_spacing) - sz0_sub)
            local_iz1 = min(n_sub - 1, int((sz + sr - zmin) / sub_spacing) - sz0_sub)
            if local_iz0 > local_iz1:
                continue  # sphere doesn't touch this z-slice

            ix0 = max(0,      int((sx - sr - xmin) / sub_spacing))
            ix1 = min(SNx-1,  int((sx + sr - xmin) / sub_spacing))
            iy0 = max(0,      int((sy - sr - ymin) / sub_spacing))
            iy1 = min(SNy-1,  int((sy + sr - ymin) / sub_spacing))
            if ix0 > ix1 or iy0 > iy1:
                continue

            llz = lz[local_iz0:local_iz1+1]
            lx  = scx[ix0:ix1+1]
            ly  = scy[iy0:iy1+1]

            dist2 = (
                (llz[:, None, None] - sz) ** 2 +
                (ly[None, :, None]  - sy) ** 2 +
                (lx[None, None, :]  - sx) ** 2
            )
            sub_slice[local_iz0:local_iz1+1, iy0:iy1+1, ix0:ix1+1] |= (dist2 <= sr ** 2)

        # Downsample: (n_sub, Ny*n_sub, Nx*n_sub) → sum over sub-axes → (Ny, Nx)
        counts = sub_slice.reshape(n_sub, Ny, n_sub, Nx, n_sub).sum(axis=(0, 2, 4))
        grid[k] = np.where(counts >= min_count, solid_label, pore_label).astype(np.int8)

    print(f"  Z-slice {Nz}/{Nz} (100%)")
    return grid, Nx, Ny, Nz


def _process_slice_batch(args):
    """Multiprocessing worker: process a contiguous batch of Z-slices (numpy fallback)."""
    (k_start, k_end,
     sx_arr, sy_arr, sz_arr, sr_arr,
     k_min_arr, k_max_arr,
     ix0_arr, ix1_arr, iy0_arr, iy1_arr,
     scx, scy, scz,
     sub_spacing, SNx, SNy, n_sub, Ny, Nx,
     zmin, min_count, solid_label, pore_label) = args

    chunk = np.empty((k_end - k_start, Ny, Nx), dtype=np.int8)
    for k_local, k in enumerate(range(k_start, k_end)):
        sub_slice = np.zeros((n_sub, SNy, SNx), dtype=bool)
        sz0_sub = k * n_sub
        for si in np.where((k_min_arr <= k) & (k_max_arr >= k))[0]:
            sx = sx_arr[si]; sy = sy_arr[si]; sz = sz_arr[si]; sr = sr_arr[si]
            local_iz0 = max(0, int((sz - sr - zmin) / sub_spacing) - sz0_sub)
            local_iz1 = min(n_sub - 1, int((sz + sr - zmin) / sub_spacing) - sz0_sub)
            if local_iz0 > local_iz1:
                continue
            ix0 = ix0_arr[si]; ix1 = ix1_arr[si]
            iy0 = iy0_arr[si]; iy1 = iy1_arr[si]
            if ix0 > ix1 or iy0 > iy1:
                continue
            lz = scz[sz0_sub + local_iz0: sz0_sub + local_iz1 + 1]
            dist2 = (
                (lz[:, None, None]                 - sz) ** 2 +
                (scy[iy0:iy1 + 1][None, :, None]   - sy) ** 2 +
                (scx[ix0:ix1 + 1][None, None, :]   - sx) ** 2
            )
            sub_slice[local_iz0:local_iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1] |= (dist2 <= sr ** 2)
        counts = sub_slice.reshape(n_sub, Ny, n_sub, Nx, n_sub).sum(axis=(0, 2, 4))
        chunk[k_local] = np.where(counts >= min_count, solid_label, pore_label).astype(np.int8)
    return k_start, chunk


def voxelize_coverage_fast(df, xmin, xmax, ymin, ymax, zmin, zmax, spacing,
                            solid_label=0, pore_label=1, n_sub=4, threshold=0.5):
    """
    Optimised drop-in replacement for voxelize_coverage. Same signature and output.

    Improvements over the baseline:
      1. Sphere data extracted as numpy arrays (no .iterrows()).
      2. Per-slice sphere pre-filtering via precomputed k_min / k_max arrays,
         so each Z-slice only touches spheres that actually overlap it.
      3a. numba (if installed): JIT-compiled loops with prange parallelism over
          Z-slices — effectively free multi-core parallelism with no pickling.
      3b. multiprocessing fallback (if numba unavailable): Z-slices divided
          across all CPU cores via concurrent.futures.ProcessPoolExecutor.
    """
    Nx = max(1, int(np.ceil((xmax - xmin) / spacing)))
    Ny = max(1, int(np.ceil((ymax - ymin) / spacing)))
    Nz = max(1, int(np.ceil((zmax - zmin) / spacing)))

    print(f"Grid: Nx={Nx}  Ny={Ny}  Nz={Nz}  ({Nx*Ny*Nz:,} voxels)")

    sub_spacing = spacing / n_sub
    SNx = Nx * n_sub
    SNy = Ny * n_sub
    SNz = Nz * n_sub

    backend = "numba" if _NUMBA_AVAILABLE else "multiprocessing"
    print(f"Coverage mode (fast/{backend}): {n_sub}^3={n_sub**3} sub-samples/voxel, "
          f"threshold={threshold:.0%}")

    # --- 1. Extract sphere data as contiguous numpy arrays (no .iterrows()) --
    sx_arr = df["X"].to_numpy(dtype=np.float64)
    sy_arr = df["Y"].to_numpy(dtype=np.float64)
    sz_arr = df["Z"].to_numpy(dtype=np.float64)
    sr_arr = df["R"].to_numpy(dtype=np.float64)

    # Sub-voxel centre coordinates (identical to baseline)
    scx = xmin + (np.arange(SNx) + 0.5) * sub_spacing
    scy = ymin + (np.arange(SNy) + 0.5) * sub_spacing
    scz = zmin + (np.arange(SNz) + 0.5) * sub_spacing

    min_count = int(np.ceil(threshold * n_sub ** 3))

    # --- 2. Pre-compute Z-slice and sub-voxel XY bounding boxes per sphere --
    # Uses floor to match the original int() truncation for positive values.
    k_min_arr = np.maximum(0,       np.floor((sz_arr - sr_arr - zmin) / spacing).astype(np.int64))
    k_max_arr = np.minimum(Nz - 1,  np.floor((sz_arr + sr_arr - zmin) / spacing).astype(np.int64))

    ix0_arr = np.maximum(0,       np.floor((sx_arr - sr_arr - xmin) / sub_spacing).astype(np.int64))
    ix1_arr = np.minimum(SNx - 1, np.floor((sx_arr + sr_arr - xmin) / sub_spacing).astype(np.int64))
    iy0_arr = np.maximum(0,       np.floor((sy_arr - sr_arr - ymin) / sub_spacing).astype(np.int64))
    iy1_arr = np.minimum(SNy - 1, np.floor((sy_arr + sr_arr - ymin) / sub_spacing).astype(np.int64))

    # --- 3a. Numba parallel kernel (preferred) --------------------------------
    if _NUMBA_AVAILABLE:
        print("  Compiling / running numba kernel …")
        grid = _coverage_kernel_nb(
            sx_arr, sy_arr, sz_arr, sr_arr,
            k_min_arr, k_max_arr,
            ix0_arr, ix1_arr, iy0_arr, iy1_arr,
            scx, scy, scz,
            sub_spacing, SNx, SNy, n_sub, Nz, Ny, Nx,
            zmin, min_count,
            np.int8(solid_label), np.int8(pore_label),
        )
        return grid, Nx, Ny, Nz

    # --- 3b. Multiprocessing fallback ----------------------------------------
    from concurrent.futures import ProcessPoolExecutor
    n_workers = os.cpu_count() or 4
    chunk_size = max(1, (Nz + n_workers - 1) // n_workers)
    batches = [(k, min(k + chunk_size, Nz)) for k in range(0, Nz, chunk_size)]
    print(f"  Multiprocessing: {n_workers} workers, {len(batches)} batches")

    common = (sx_arr, sy_arr, sz_arr, sr_arr,
              k_min_arr, k_max_arr,
              ix0_arr, ix1_arr, iy0_arr, iy1_arr,
              scx, scy, scz,
              sub_spacing, SNx, SNy, n_sub, Ny, Nx,
              zmin, min_count, solid_label, pore_label)

    grid = np.empty((Nz, Ny, Nx), dtype=np.int8)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        args_list = [(ks, ke) + common for ks, ke in batches]
        for k_start, chunk in executor.map(_process_slice_batch, args_list):
            grid[k_start: k_start + chunk.shape[0]] = chunk

    return grid, Nx, Ny, Nz


def save_slice(slice_2d, axis, index, output_base, solid_label, out_dir):
    """Save a single 2D cross-section as a grayscale PNG.

    solid_label is rendered black, pore is white.
    axis: 'x' (YZ plane), 'y' (XZ plane), or 'z' (XY plane).
    """
    # Normalise to [0, 1]: solid → 0 (black), pore → 1 (white)
    img = np.where(slice_2d == solid_label, 0.0, 1.0)

    axis_labels = {
        "x": ("Y index", "Z index"),
        "y": ("X index", "Z index"),
        "z": ("X index", "Y index"),
    }
    xlabel, ylabel = axis_labels[axis]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cross-section {axis.upper()}={index}")
    fig.tight_layout()

    path = os.path.join(out_dir, f"{output_base}_slice_{axis}{index:04d}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Slice written → {path}")


def save_slices(grid, Nx, Ny, Nz, args, out_dir):
    """Export all requested cross-section PNGs.

    grid shape: (Nz, Ny, Nx)
    --slice-x I  →  grid[:, :, I]  (YZ plane, rows=Z, cols=Y)
    --slice-y J  →  grid[:, J, :]  (XZ plane, rows=Z, cols=X)
    --slice-z K  →  grid[K, :, :]  (XY plane, rows=Y, cols=X)
    """
    def resolve(indices, size):
        """None (not specified) → middle; empty list → middle; otherwise use given indices."""
        if indices is None or len(indices) == 0:
            return [size // 2]
        return indices

    for i in resolve(args.slice_x, Nx):
        if not (0 <= i < Nx):
            print(f"Warning: slice-x={i} out of range [0, {Nx-1}], skipped.")
            continue
        save_slice(grid[:, :, i], "x", i, args.output, args.solid_label, out_dir)

    for j in resolve(args.slice_y, Ny):
        if not (0 <= j < Ny):
            print(f"Warning: slice-y={j} out of range [0, {Ny-1}], skipped.")
            continue
        save_slice(grid[:, j, :], "y", j, args.output, args.solid_label, out_dir)

    for k in resolve(args.slice_z, Nz):
        if not (0 <= k < Nz):
            print(f"Warning: slice-z={k} out of range [0, {Nz-1}], skipped.")
            continue
        save_slice(grid[k, :, :], "z", k, args.output, args.solid_label, out_dir)


def write_db(path, Nx, Ny, Nz, solid_label, pore_label, raw_filename, spacing):
    """Write a minimal LBPM .db domain configuration file."""
    content = f"""\
Domain {{
    Filename = "{raw_filename}"
    ReadType = "8bit"
    N = {Nx}, {Ny}, {Nz}
    nproc = 1, 1, 1
    n = {Nx}, {Ny}, {Nz}
    voxel_length = {spacing}
    ReadValues  = {solid_label}, {pore_label}
    WriteValues = {solid_label}, {pore_label}
    BC = 0
}}
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Config written → {path}")


def main():
    args = parse_args()

    # --- Load data ---
    df = pd.read_csv(args.csv)
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    required = {"R", "X", "Y", "Z"}
    if not required.issubset(df.columns):
        sys.exit(f"CSV must have columns R, X, Y, Z. Found: {list(df.columns)}")

    print(f"Loaded {len(df)} spheres from '{args.csv}'")

    # --- Resolve bounds ---
    xmin = args.xmin if args.xmin is not None else auto_bounds(df, "X")[0]
    xmax = args.xmax if args.xmax is not None else auto_bounds(df, "X")[1]
    ymin = args.ymin if args.ymin is not None else auto_bounds(df, "Y")[0]
    ymax = args.ymax if args.ymax is not None else auto_bounds(df, "Y")[1]
    zmin = args.zmin if args.zmin is not None else auto_bounds(df, "Z")[0]
    zmax = args.zmax if args.zmax is not None else auto_bounds(df, "Z")[1]

    print(f"Domain bounds:")
    print(f"  X: [{xmin:.4f}, {xmax:.4f}]")
    print(f"  Y: [{ymin:.4f}, {ymax:.4f}]")
    print(f"  Z: [{zmin:.4f}, {zmax:.4f}]")
    print(f"Voxel spacing: {args.spacing}")

    # --- Voxelize ---
    if args.mode == "coverage":
        fn = voxelize_coverage_fast if args.fast else voxelize_coverage
        grid, Nx, Ny, Nz = fn(
            df, xmin, xmax, ymin, ymax, zmin, zmax, args.spacing,
            solid_label=args.solid_label, pore_label=args.pore_label,
            n_sub=args.sub_samples, threshold=args.threshold,
        )
    else:
        grid, Nx, Ny, Nz = voxelize(
            df, xmin, xmax, ymin, ymax, zmin, zmax, args.spacing,
            solid_label=args.solid_label, pore_label=args.pore_label,
        )

    # --- Statistics ---
    n_solid = int(np.sum(grid == args.solid_label))
    n_pore  = int(np.sum(grid == args.pore_label))
    total   = Nx * Ny * Nz
    print(f"Solid voxels: {n_solid:,} ({100*n_solid/total:.1f}%)")
    print(f"Pore  voxels: {n_pore:,}  ({100*n_pore/total:.1f}%)")

    # --- Output directory: output/<YYYY-MM-DD_HH-MM-SS>/ ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("output", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}/")

    # --- Write output ---
    raw_name = f"{args.output}.raw"
    raw_path = os.path.join(out_dir, raw_name)
    db_path  = os.path.join(out_dir, f"{args.output}.db")

    grid.tofile(raw_path)
    print(f"Geometry written → {raw_path}")

    voxel_length = args.voxel_length if args.voxel_length is not None else args.spacing
    write_db(db_path, Nx, Ny, Nz, args.solid_label, args.pore_label,
             raw_name, voxel_length)

    # --- Save arguments ---
    args_path = os.path.join(out_dir, "args.txt")
    with open(args_path, "w") as f:
        f.write(f"Command: python3 voxelize.py {' '.join(sys.argv[1:])}\n\n")
        for key, val in sorted(vars(args).items()):
            f.write(f"{key}: {val}\n")
        f.write(f"\n# Resolved values\n")
        f.write(f"xmin: {xmin}  xmax: {xmax}\n")
        f.write(f"ymin: {ymin}  ymax: {ymax}\n")
        f.write(f"zmin: {zmin}  zmax: {zmax}\n")
        f.write(f"voxel_length (um): {voxel_length}\n")
        f.write(f"Nx: {Nx}  Ny: {Ny}  Nz: {Nz}\n")
        f.write(f"solid_fraction: {100*n_solid/total:.2f}%\n")
        f.write(f"pore_fraction:  {100*n_pore/total:.2f}%\n")
    print(f"Arguments saved → {args_path}")

    # --- Cross-section images ---
    if not args.no_slices:
        save_slices(grid, Nx, Ny, Nz, args, out_dir)


if __name__ == "__main__":
    main()
