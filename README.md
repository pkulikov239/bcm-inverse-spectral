# Boundary control method for an inverse spectral problem

[![DOI](https://zenodo.org/badge/1387449121.svg)](https://doi.org/10.5281/zenodo.22961134)

Code and numerical experiments for the paper

> A. Hannukainen, P. Kulikov, M. Lassas, L. Oksanen,
> *An inverse spectral problem for the wave equation with a varying Robin parameter at the boundary*.
> Preprint, 2026.

The wave speed $c$ of a one-dimensional wave equation on $(0,1)$ is reconstructed from two
spectra: the eigenvalues of a Neumann–Dirichlet problem and of a Robin–Dirichlet problem with
a small impedance $\omega$. The reconstruction uses the boundary control method.

## Method

1. **Spectral data.** The eigenvalues $\lambda_n(0)$ and $\lambda_n(\omega)$, $n = 1, \dots, L$, are
   simulated by the finite element method (FEM). The true $c$ is used only here and for scoring.
2. **Observation time.** The travel time $\tau(1)$ is estimated from $\lambda_n(0)$ by the Weyl
   asymptotics, which fixes the observation time $T$.
3. **Neumann-to-Dirichlet map.** The truncated map $\Lambda_L^\omega$ is built from the two spectra.
4. **Boundary control.** The Blagoveščenskiĭ operator $K$ is assembled from $\Lambda_L^\omega$, and
   regularized control problems give the volumes $V(r)$.
5. **Wave speed.** $c$ is recovered from the derivative of $V$.

## Contents

| File | Purpose |
|---|---|
| `forward.py` | Finite element method (FEM) simulation of the spectral data and construction of the truncated Neumann-to-Dirichlet map |
| `bcm.py` | Boundary control method: from the truncated Neumann-to-Dirichlet map to the volumes $V(r)$ |
| `wavespeed.py` | Reconstruction of the wave speed from $V$ |
| `numerical_results.ipynb` | All figures and tables of Section 7 of the paper |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT license |

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Reproducing the results

```
jupyter notebook numerical_results.ipynb
```

Run all cells. The notebook takes a few minutes on a laptop and shows all figures and tables
inline; nothing is written to disk. It contains:

- **Section 7.1:** boundary sources for $c \equiv 1$, checked with an independent finite difference solver;
- **Section 7.2:** reconstructions of six wave speed profiles and their errors;
- the choice of the margin $\eta$ in $T = (1-\eta)\hat\tau(1)$;
- **Section 7.3:** sensitivity of the reconstruction to noise in the spectral data.

The parameters of the method are chosen by a grid search against the true wave speed, so the
results show the accuracy attainable with the method.

The results in the paper were obtained with version 1.0 of this repository,
[doi:10.5281/zenodo.22961135](https://doi.org/10.5281/zenodo.22961135).

## Citation

If you use this code, please cite the paper above and the archived code:

> A. Hannukainen, P. Kulikov, M. Lassas, L. Oksanen, *Code for "An inverse spectral problem for
> the wave equation with a varying Robin parameter at the boundary"*, Zenodo,
> [doi:10.5281/zenodo.22961134](https://doi.org/10.5281/zenodo.22961134).

```bibtex
@software{bcm_inverse_spectral,
  author    = {Hannukainen, Antti and Kulikov, Petr and Lassas, Matti and Oksanen, Lauri},
  title     = {Code for "An inverse spectral problem for the wave equation with a varying Robin parameter at the boundary"},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22961134},
  url       = {https://github.com/pkulikov239/bcm-inverse-spectral}
}
```

## License

MIT, see `LICENSE`.
