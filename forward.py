"""forward.py — the forward map  c ↦ {λ_n(0), λ_n(ω)} ↦ Λ.

`solve_wave` integrates the wave equation (𝒲_f) by the method of lines: a
second-difference operator in x, `scipy.integrate.solve_ivp` in t.  `spectral`
solves the impedance spectral problem (𝒮_ω) by P1 finite elements (scikit-fem)
and `Λ_map` turns its two spectra into the Neumann-to-Dirichlet map.  All
dependence on ω sits in the boundary matrix G, so B_0, G and M are assembled
once and every ω reuses them.  CPU only.

    Ω = (0,1)              the medium; c ≡ 1 outside it
    c, C_max               wave speed and its bound
    t, x, Δx               the time and space grids
    f                      Neumann datum at x = 0;  ∂ₓu(t,0) = f(t)

    (𝒲_f)                  ∂ₜ²u = c²∂ₓ²u,  ∂ₓu(t,0) = f(t),  u(t,1) = 0,
                           u(0,·) = ∂ₜu(0,·) = 0
    A, e_0                 ü = Au + e_0 f(t), the semi-discrete form of (𝒲_f)

    (𝒮_ω)                  −c²∂ₓ²φ = λφ,  ∂ₓφ(0) + ωφ(0) = 0,  φ(1) = 0,  in Ω
    ω, L, N_x              impedance, modes kept, and the mesh h = 1/N_x
    B_ω = B_0 − ω G        stiffness;  G = ∫_{x=0} uv  is the Robin term
    M                      the mass matrix of (·,·)_{L²_c}
    λ_0, λ_ω, λ̇_0          the two spectra and λ′(0) ≈ (λ(ω) − λ(0))/ω
    φ, φ_h                 eigenfunctions
    s_λ(τ)                 the kernel  s_λ(τ) = sin(√λ τ)/√λ
    S_λ[f]                 the convolution operator
                           S_λ[f](t) = ∫_0^t s_λ(t − s) f(s) ds = (s_λ ∗ f)(t)
    Λ[f] = u^f|_{x=0}      the map;  Λ_L^ω  its truncation,
                           Λ_L^ω[f] = Σ_{n≤L} λ̇_n S_{λ_n(0)}[f]

Both routes to Λ live on Ω̄ = [0,1] and both carry the Dirichlet condition
u(t,1) = 0, so (𝒮_ω) is the exact separation of variables of (𝒲_f) and Λ_L^ω is
a truncation of the series of Proposition 1, not an approximation valid only on
a window.  Nothing here evaluates c outside Ω.  The two routes share no
discretisation either — finite differences and an adaptive Runge–Kutta on one
side, P1 elements and an eigenvalue solve on the other — so comparing them is a
genuine check.

Λ is the map for the datum as (𝒲_f) states it, ∂ₓu(t,0) = f(t), not the outward
conormal ∂_ν = −∂ₓ.  That orientation is what fixes the order of the terms in
the Blagoveščenskiĭ operator K assembled in bcm.
"""

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.signal import fftconvolve

from skfem import Basis, BilinearForm, ElementLineP1, FacetBasis, MeshLine, asm
from skfem.models.poisson import laplace, mass

__all__ = ["x_grid", "solve_wave", "spectral", "spectral_data", "S_λ",
           "kernel_from_data", "Λ_kernel", "Λ_map",
           "travel_time_from_spectrum"]


# =============================================================================
# The wave equation (𝒲_f)
# =============================================================================

def x_grid(t, C_max):
    """A grid on Ω̄ whose resolution matches the time mesh, Δx ≈ C_max Δt.

    The method of lines has no CFL condition, so this is a choice of accuracy
    rather than a constraint; it is the resolution the explicit scheme used to
    need, kept so that Δx and Δt refine together."""
    return np.linspace(0, 1, int(np.floor(1 / ((t[1] - t[0])*C_max))) + 1)


def _operator(x, c):
    """(A, e_0) of the semi-discrete problem  ü = A u + e_0 f(t)  on the
    interior nodes x_0,…,x_{n−1}, the Dirichlet node x_n = 1 being dropped.

    A = diag(c²)·L/Δx² with L the second difference.  The Neumann condition
    enters through the ghost value u_{−1} = u_1 − 2Δx f, which doubles the
    superdiagonal entry of the first row and puts the datum in e_0."""
    n, Δx = len(x) - 1, x[1] - x[0]

    L = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil")
    L[0, 1] = 2.0
    A = sp.diags(c(x[:n])**2 / Δx**2) @ L.tocsr()

    e_0 = np.zeros(n)
    e_0[0] = -2*c(x[0])**2 / Δx
    return A.tocsr(), e_0


def solve_wave(t, x, c, f, rtol=1e-8, atol=1e-10):
    """The solution of (𝒲_f) on the grid t × x, as a (len(t), len(x)) array.

    Writing y = (u, ∂ₜu) turns the semi-discrete system into a first-order ODE
    that `solve_ivp` integrates with the explicit Runge–Kutta DOP853; the
    default tolerances put the time error well below the O(Δx²) error of the
    spatial difference, so refining x alone converges.

    `max_step` is the sampling interval of t, and it is not optional.  The
    controls of the boundary control method are supported in [T−r, T], so the
    solver starts from y = 0 with an RHS that vanishes identically until the
    control switches on.  Its error estimate is then exactly 0, the step size
    grows without bound, and an unconstrained integrator steps straight over
    the support of f and returns u ≡ 0.  Capping the step at Δt guarantees the
    control is sampled wherever the caller asked to see the solution."""
    A, e_0 = _operator(x, c)
    n = len(e_0)

    def rhs(s, y):
        return np.concatenate((y[n:], A @ y[:n] + e_0*f(s)))

    sol = solve_ivp(rhs, (t[0], t[-1]), np.zeros(2*n), t_eval=t,
                    method="DOP853", rtol=rtol, atol=atol,
                    max_step=t[1] - t[0])
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    u = np.zeros((len(t), len(x)))          # the Dirichlet node stays 0
    u[:, :n] = sol.y[:n].T
    return u


# =============================================================================
# The impedance spectral problem (𝒮_ω)
# =============================================================================

def spectral(c, ω=(0.0,), L=150, N_x=600):
    """The L lowest eigenpairs of (𝒮_ω) on Ω, for each ω, by P1 finite elements.

    (𝒮_ω) is posed on Ω = (0,1) exactly as stated: the Robin condition
    ∂ₓφ(0) + ωφ(0) = 0 sits at x = 0 and the Dirichlet condition φ(1) = 0 at
    x = 1, the same condition (𝒲_f) carries.  The mesh is the uniform partition
    x_k = k h of Ω̄ with h = 1/N_x, and c is never evaluated outside Ω.

    Ritz–Galerkin on  V_h = span{φ̂_0,…,φ̂_{N_x−1}}  gives  B_ω α = λ_h M α  with
    B_ω(u,v) = ∫u′v′ − ω u(0)v(0)  and  (u,v)_{L²_c} = ∫ u v c⁻².  The Robin
    term is the facet mass G, whose one nonzero entry is G[0,0] = 1, so
    B_ω = B_0 − ω G and the three matrices are assembled once and shared.

    skfem's `intorder=2` is exactly the two-point Gauss–Legendre rule
    ξ = ±1/√3, w = 1 that the paper prescribes for M.  `subset_by_index` asks
    LAPACK for only the L lowest pairs.

    Returns (x, λ, φ_h) of shapes (N_x+1,), (n_ω, L) and (n_ω, N_x+1, L)."""
    mesh = MeshLine(np.linspace(0.0, 1.0, N_x + 1))      # x_k = k h,  h = 1/N_x
    basis = Basis(mesh, ElementLineP1(), intorder=2)
    tol = 1e-12
    rim = FacetBasis(mesh, ElementLineP1(),
                     facets=mesh.facets_satisfying(lambda x: x[0] < tol))

    @BilinearForm
    def mass_c(u, v, w):
        return u * v / c(w.x[0])**2                     # (·,·)_{L²_c}

    # active dofs: every node but the Dirichlet one at x = 1
    dofs = basis.complement_dofs(basis.get_dofs(lambda x: x[0] > 1.0 - tol))
    L = min(L, len(dofs))                               # at most one mode per dof
    cut = lambda A: A.toarray()[np.ix_(dofs, dofs)]
    B_0, G, M = cut(asm(laplace, basis)), cut(asm(mass, rim)), cut(asm(mass_c, basis))

    λ, φ_h = [], []
    for ω_i in np.atleast_1d(ω).astype(float):
        λ_i, α = eigh(B_0 - ω_i*G, M, subset_by_index=[0, L - 1])
        φ = np.zeros((N_x + 1, L))                      # Dirichlet node stays 0
        φ[dofs, :] = α
        λ.append(λ_i)
        φ_h.append(φ)

    return mesh.p[0], np.array(λ), np.array(φ_h)


def S_λ(λ_0, t):
    """The kernels s_{λ_n(0)} of the convolution operators S_{λ_n(0)}, sampled on t,

        s_λ(τ) = sin(√λ τ)/√λ,     S_λ[f](t) = ∫_0^t s_λ(t − s) f(s) ds = (s_λ ∗ f)(t).

    Returns the array of values s_{λ_n(0)}(t_j), shape (len(λ_0), len(t));
    the operator S_λ itself is applied by convolution with these kernels."""
    root_λ = np.sqrt(λ_0)[:, None]
    return np.sin(root_λ * t[None, :]) / root_λ


def spectral_data(c, L=150, ω=1e-4, N_x=600):
    """The data of the inverse problem, simulated by (𝒮_ω):  (λ_0, λ̇_0) with
    λ_0 = {λ_n(0)} and λ̇_0 = {(λ_n(ω) − λ_n(0))/ω} ≈ {λ_n′(0)},  n = 1, …, L."""
    _, (λ_ω, λ_0), _ = spectral(c, (ω, 0.0), L, N_x)
    return λ_0, (λ_ω - λ_0) / ω


def kernel_from_data(t, λ_0, λ̇_0):
    """The kernel of Λ_L^ω from the spectral data alone, sampled on t,

        k(τ) = Σ_n λ̇_n s_{λ_n(0)}(τ),   so that   Λ_L^ω[f] = Σ_n λ̇_n S_{λ_n(0)}[f] = k ∗ f."""
    return np.asarray(λ̇_0) @ S_λ(np.asarray(λ_0), t)


def Λ_kernel(t, c, L=150, ω=1e-4, N_x=600):
    """The convolution kernel of Λ_L^ω sampled on t,

        k(τ) = Σ_{n=1}^{L} (λ_n(ω) − λ_n(0))/ω · s_{λ_n(0)}(τ),

    so that  Λ_L^ω[f] = Σ_n (λ_n(ω) − λ_n(0))/ω · S_{λ_n(0)}[f] = k ∗ f.
    Returned together with λ_0."""
    λ_0, λ̇_0 = spectral_data(c, L, ω, N_x)
    return kernel_from_data(t, λ_0, λ̇_0), λ_0


def Λ_map(t, c, f, L=150, ω=1e-4, N_x=600):
    """The discrete truncated Neumann-to-Dirichlet map applied to f,

        Λ_L^ω[f] = Σ_{n=1}^{L}  (λ_n(ω) − λ_n(0))/ω · S_{λ_n(0)}[f],

    where S_{λ_n(0)}[f] = s_{λ_n(0)} ∗ f is the convolution operator and the
    quotient stands for λ̇_0 = λ′_n(0) = −|φ_n(0)|².  Sampled on t.

    All L convolutions are carried out at once, as one convolution of f with the
    kernel k = Σ_n λ̇_n s_{λ_n(0)}.  The convolution is the rectangle rule on t;
    since k(0) = 0 it coincides with the trapezoidal rule whenever f(0) = 0."""
    k, _ = Λ_kernel(t, c, L, ω, N_x)
    f_t = f(t)
    return fftconvolve(k, f_t, mode="full")[:len(f_t)] * (t[1] - t[0])


def travel_time_from_spectrum(λ_0, n_range=(5, 40)):
    """τ(1) estimated from the data alone, by the Weyl asymptotics of (𝒮_0),

        √λ_n(0) = π(n − ½)/τ(1) + a + b/n + o(1/n),

    fitted by least squares on n ∈ n_range; τ̂(1) = π / slope.  The constant a
    absorbs the discretisation error of the eigenvalues (a = 0 for the exact
    spectrum).  The true c is never used.  On synthetic P1 spectra the error is
    dominated by the mesh error of (𝒮_ω), O(h²), and is an underestimate
    (≈ −0.3% at N_x = 600), which is the safe side: r never enters the range
    r > τ(1)."""
    λ_0 = np.asarray(λ_0, float)
    lo, hi = n_range[0], min(n_range[1], len(λ_0))
    n = np.arange(lo, hi + 1)
    A = np.column_stack([n, np.ones_like(n, dtype=float), 1.0/n])
    slope = np.linalg.lstsq(A, np.sqrt(λ_0[n - 1]), rcond=None)[0][0]
    return float(np.pi / slope)