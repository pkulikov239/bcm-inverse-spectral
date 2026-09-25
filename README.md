# Boundary control method for an inverse spectral problem

Code and numerical experiments for the paper

> A. Hannukainen, <your name>, M. Lassas, L. Oksanen,
> *An inverse spectral problem for the wave equation with a varying Robin parameter at the boundary*,
> <arXiv link or journal>.

The wave speed of a one-dimensional wave equation is reconstructed from two spectra,
the eigenvalues of a Neumann–Dirichlet problem and of a Robin–Dirichlet problem with a
small impedance ω, by the boundary control method.

## Contents

| File | Purpose |
|---|---|
| `forward.py` | Simulation of the spectral data by finite elements, and the Neumann-to-Dirichlet map |
| `bcm.py` | Boundary control method: from the spectra to the volumes V(r) |
| `wavespeed.py` | Reconstruction of the wave speed from V |
| `numerical_results.ipynb` | All figures and tables of Section 7 of the paper |

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Reproducing the results

```
jupyter notebook numerical_results.ipynb
```

Run all cells. The notebook takes a few minutes on a laptop and shows all figures and
tables inline. The results in the paper were obtained with version 1.0 of this repository.

## Citation

If you use this code, please cite the paper above and the archived code:

> <authors>, *Code for "An inverse spectral problem for the wave equation with a varying
> Robin parameter at the boundary"*, Zenodo, <DOI>.

## License

MIT, see `LICENSE`.
