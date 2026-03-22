# Spherical Packing Voxelizer

Converts DEM sphere packing data (CSV) into a binary voxel geometry file ready for fluid flow simulation with **LBPM (Lattice Boltzmann for Porous Media)**.

- LBPM repository: https://github.com/OPM/LBPM
- LBPM documentation: https://lbpm-sim.org

## Output

Each run creates a timestamped folder `output/YYYY-MM-DD_HH-MM-SS/` containing:

| File | Description |
|------|-------------|
| `geometry.raw` | Raw binary geometry (`int8`, shape `Nz×Ny×Nx`): `0` = solid, `1` = pore |
| `geometry.db` | LBPM domain configuration file |
| `geometry_slice_x****.png` | YZ cross-section at the specified X index |
| `geometry_slice_y****.png` | XZ cross-section at the specified Y index |
| `geometry_slice_z****.png` | XY cross-section at the specified Z index |

---

## Installation

> **Python version:** Python 3.9 or newer is required. The `numba` dependency (used for the `--fast` acceleration) does not support Python 3.6/3.7/3.8.

### On NCI Gadi (HPC)

```bash
# 1. Load a Python 3.9+ module
module load python3/3.12.1

# 2. Confirm the version
python3.12 --version

# 3. Create a virtual environment using the module's Python
python3.12 -m venv .venv

# 4. Activate it
source .venv/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### On a standard Linux / macOS system

**Linux (RHEL/CentOS):**
```bash
sudo yum install python3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install python3 python3-venv
```

**macOS:**
```bash
brew install python
```

```bash
cd spherical-packing-voxelizer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### On Windows

1. Download the installer from https://www.python.org/downloads/windows/
2. Run the installer — **check "Add Python to PATH"** before clicking Install
3. Open Command Prompt or PowerShell in the project folder:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

> **PowerShell note:** if you get an error about execution policy, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Usage

```bash
python3 voxelize.py <csv_file> [options]
```

### Minimal example

```bash
# Linux / macOS
python3 voxelize.py data/ball_data.csv --spacing 0.1

# Windows
python voxelize.py data/ball_data.csv --spacing 0.1
```

### Full example

```bash
python3 voxelize.py data/ball_data.csv \
    --spacing 0.05 \
    --xmin -5 --xmax 5 \
    --ymin -5 --ymax 5 \
    --zmin -5 --zmax 5 \
    --mode coverage --sub-samples 4 --threshold 0.5 \
    --output geometry
```

---

## Arguments

### Input

| Argument | Description |
|----------|-------------|
| `csv` | Path to the input CSV file. Required columns: `ID`, `R`, `X`, `Y`, `Z` |

### Domain bounds

If not specified, bounds are auto-detected from the data with a margin of one maximum sphere radius.

| Argument | Description |
|----------|-------------|
| `--xmin`, `--xmax` | X axis bounds |
| `--ymin`, `--ymax` | Y axis bounds |
| `--zmin`, `--zmax` | Z axis bounds |
| `--spacing`, `-s` | Voxel size in the same units as the coordinates (default: `0.1`) |
| `--voxel-length` | Physical voxel size written to the `.db` file, in **micrometres (µm)**. Defaults to `--spacing` if not set. Use this when your coordinates are not in µm (e.g. if coords are in metres, pass `--voxel-length <spacing × 1e6>`). |

### Voxelization mode

| Argument | Description |
|----------|-------------|
| `--mode` | Voxelization method: `center` (default) or `coverage` |
| `--fast` | Use the optimised coverage implementation (see below). Only applies to `--mode coverage`. |

#### `center` mode (default, fast)

A voxel is marked **solid** if its centre point lies inside any sphere. Fast and memory-efficient, but the solid/pore boundary can be blocky at coarse resolutions.

#### `coverage` mode (accurate)

A voxel is marked **solid** if the fraction of its volume covered by sphere(s) is greater than or equal to `--threshold`. Internally, each voxel is subdivided into `N³` sub-voxels (controlled by `--sub-samples`) and the fraction of sub-voxels inside any sphere is computed. Processes one Z-slice at a time so memory usage stays low regardless of grid size.

| Argument | Default | Description |
|----------|---------|-------------|
| `--sub-samples N` | `4` | Sub-divisions per voxel axis (`N³` samples per voxel). Higher = more accurate, more compute. |
| `--threshold F` | `0.5` | Minimum solid volume fraction to classify a voxel as solid (0–1). |
| `--fast` | off | Enables `voxelize_coverage_fast`: pre-filters spheres per Z-slice and parallelises across all CPU cores. If `numba` is installed the inner loops are JIT-compiled (fastest); otherwise `multiprocessing` is used as a fallback. Produces identical results to the baseline. Recommended for grids larger than ~500³. |

### Output

| Argument | Description |
|----------|-------------|
| `--output`, `-o` | Base name for output files (default: `geometry`) |

### Cross-section images

By default the middle slice of each axis (X, Y, Z) is exported as a PNG. Solid grains are **black**, pore space is **white**.

| Argument | Description |
|----------|-------------|
| `--slice-x [I ...]` | Voxel index/indices along X to export as YZ-plane PNGs. Omit indices for middle slice. |
| `--slice-y [J ...]` | Voxel index/indices along Y to export as XZ-plane PNGs. Omit indices for middle slice. |
| `--slice-z [K ...]` | Voxel index/indices along Z to export as XY-plane PNGs. Omit indices for middle slice. |
| `--no-slices` | Disable PNG output entirely. |

### Labels

| Argument | Default | Description |
|----------|---------|-------------|
| `--solid-label` | `0` | Byte value written for solid voxels |
| `--pore-label` | `1` | Byte value written for pore/fluid voxels |

---

## LBPM integration

The generated `geometry.db` uses the `Domain` block format expected by LBPM:

```
Domain {
    Filename = "geometry.raw"
    ReadType = "8bit"
    N = 100, 100, 100
    nproc = 1, 1, 1
    n = 100, 100, 100
    voxel_length = 0.1
    ReadValues  = 0, 1
    WriteValues = 0, 1
    BC = 0
}
```

- `voxel_length` is set to `--voxel-length` if provided, otherwise falls back to `--spacing`. It is interpreted by LBPM in **micrometres (µm)**. If your coordinates are not in µm, use `--voxel-length` to set the correct physical voxel size without affecting the voxelization grid.
- `BC = 0` means periodic boundaries. With a body force (`F` in the `BGK` block), flow is driven along the specified axis.
- The Z axis is the primary flow direction in LBPM — if your packing has a preferred direction, align it with Z.

### Running a single-phase simulation

To run a single-phase (BGK) flow simulation, append the following blocks to the generated `.db` file and run `lbpm_bgk_simulator`:

```
BGK {
   tau = 1.0
   F = 0.0, 0.0, 1.0e-5
   timestepMax = 200000
   tolerance = 0.01
   rho = 1.0
}

Analysis {
   analysis_interval = 1000
   subphase_analysis_interval = 50000
   N_threads = 4
   visualization_interval = 10000
   restart_interval = 1000000
   restart_file = "Restart"
}

Visualization {
   write_silo = true
   save_8bit_raw = true
   save_pressure = true
   save_velocity = true
}
```

Key parameters:

| Parameter | Units | Description |
|-----------|-------|-------------|
| `tau` | dimensionless | Relaxation time. Effective kinematic viscosity in lattice units: `ν = (tau − 0.5) / 3` |
| `F` | lu/ts² | Body force vector `(Fx, Fy, Fz)` in lattice units per timestep². Drives flow along the specified axis (here +Z) |
| `rho` | lu⁻³ | Initial fluid density in lattice units (typically 1.0) |
| `nu` | lu²/ts | Kinematic viscosity in lattice units. Note: in BGK mode `tau` takes precedence — `nu` is informational |
| `timestepMax` | timesteps | Maximum number of LBM iterations |
| `tolerance` | dimensionless | Convergence criterion on the relative change in flow rate between analysis intervals |
| `analysis_interval` | timesteps | How often flow statistics are computed and written |
| `visualization_interval` | timesteps | How often full field snapshots are saved |
| `restart_interval` | timesteps | How often a restart checkpoint is written |
| `N_threads` | — | Number of CPU threads used for analysis |

> **Physical units:** LBPM works in lattice units (lu, ts). `voxel_length` is in **micrometres** and acts as the conversion factor between lattice and physical space: permeability scales as `voxel_length²` (µm²) and velocity scales as `voxel_length / timestep_size` (µm/ts).
