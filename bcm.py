"""bcm.py — the boundary control method: Λ ↦ K ↦ f_αβr ↦ V.

`Discr` fixes the time mesh and the hat basis.  The operators are named after
the symbols they carry: `Λ`, `K`, `b`, `M`, `S`, `Q`, `f_αβr`, `V`.
Every matrix is assembled once per call — `M`, `S` and the hat-evaluation
matrix `Φ̂` come from one cached skfem basis, and `Q` builds K, M, S and b in
the single place both entry points share.  CPU only.

    T, 2T, Δt, N           the interval [0,2T], its step and half its length
    φ̂_j, Φ̂                 the 2N interior hat functions and their values
    L, ω                   truncation and impedance behind Λ_L^ω

    Λ                      the Neumann-to-Dirichlet map,  Λ[f] = u^f|_{x=0}
    B, J, R                B[f](t) = −1_{(0,T)}(t)∫_t^T f,
                           J[f](t) = ½·1_{(0,T)}(t)∫_t^{2T−t} f,  R[f](t) = f(2T − t)
    K = RΛRJ − JΛ          the Blagoveščenskiĭ operator, compact and ⪰ 0
    P_r                    the projector onto [T−r, T]: the trailing block
    M, S                   the P1 mass and stiffness matrices; M represents I,
                           and ⟨Sf,h⟩ = (∂ₜf, ∂ₜh)
    α, β                   the L² and H¹ regularisation weights
    Q = K + αM + βS        the system of the control problem, b = (φ̂, B1)
    f_αβr = Q⁻¹ P_r b      the optimal control, non-positive
    V(r) = (f_αβr, B1)     the travel-time volume  ∫_{Ω(r)} c⁻² dx
    τ(x) = ∫_0^x dy/c(y),  χ = τ⁻¹,  Ω(r) = {x : τ(x) ≤ r}

On the choice of T and of the range of r.  Everything is posed on Ω = (0,1):
(𝒲_f) carries u(t,1) = 0.  The identity (u^f(T),1)_{L²_c} = (f,B1) holds only
for supp f ⊂ [T−r,T] with r ≤ τ(1): for r > τ(1) the front has reached x = 1,
∂ₓu^f(t,1) ≠ 0, and (f,B1) is no longer the volume of anything.  τ(1) depends
on the unknown c, so it is read from the data: `InvProb.τ_1` fits the Weyl
asymptotics of the spectrum λ_n(0) (`fwd.travel_time_from_spectrum`).  `V`
returns only the controls with r ≤ τ̂(1); any T ≥ τ̂(1) is admissible, T = τ̂(1)
being the cheapest.  `ws.travel_time(c)` uses the true c and is for scoring
only.

On the discrete control space.  T = t_N + Δt/2 lies inside an element, so the
last hat φ̂_{N−1} reaches Δt/2 past T and the discrete controls lie in H(r)
only up to O(Δt).

On the orientation of B and K.  (𝒲_f) prescribes ∂ₓu(t,0) = f(t), and at the
left endpoint the boundary term enters with the opposite sign to the usual
outward-conormal convention.  That is why B integrates from T to t and why K
takes the terms in the order RΛRJ − JΛ.  With these definitions the
Blagoveščenskiĭ identities hold as usually stated,

    (u^f(T), u^h(T))_{L²_c} = (f, K h),        (u^f(T), 1)_{L²_c} = (f, B1),

K is symmetric positive semidefinite, E(f) = (f,Kf) − 2(f,B1) + ‖f‖²_{α,β} is
convex, and its minimiser solves (P_r K P_r + αI + βS) f = P_r B1.  Since
B1(t) = t − T ≤ 0 on (0,T), f_αβr is non-positive — a downward near-delta at
t = T − r — while V = (f_αβr, B1) pairs two non-positive functions and is
positive.
"""

import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, NamedTuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, solve_triangular
from scipy.signal import fftconvolve

from skfem import Basis, ElementLineP1, MeshLine, asm
from skfem.models.poisson import laplace, mass

import forward as fwd

__all__ = ["Discr", "InvProb", "B1", "u", "Λ_trace", "Λ", "K", "b", "M", "S",
           "Q", "f_αβr", "V", "V_exact",
           # ASCII aliases
           "Lambda", "Lambda_trace", "f_abr"]


# =============================================================================
# The time mesh, its hat basis, and the matrices that live on it
# =============================================================================

@lru_cache(maxsize=8)
def _fem(N, T):
    """One skfem basis on [0,2T] and everything read off it: the interior dofs
    (they are the hats φ̂_j, in order) and the mass and stiffness matrices cut
    down to the first N hats.  Cached, so nothing is assembled twice."""
    t = np.linspace(0, 2*T, 2*N + 2)
    basis = Basis(MeshLine(t), ElementLineP1(), intorder=2)

    tol = 1e-12 * max(1.0, abs(t[-1]))
    rim = basis.get_dofs(lambda x: (x[0] < tol) | (x[0] > t[-1] - tol))
    dofs = np.sort(basis.complement_dofs(rim))[:N]

    cut = lambda A: A.tocsr()[dofs][:, dofs].toarray()
    return basis, dofs, cut(asm(mass, basis)), cut(asm(laplace, basis))


@dataclass(frozen=True)
class Discr:
    """The uniform mesh on [0,2T] and the 2N interior hat functions φ̂_j.

    φ̂_j is centred at t_{j+1} and supported on [t_j, t_{j+2}].  The refined
    mesh `t_refined` carries the quadrature of the operators."""
    N: int = 50
    T: float = 1.0
    refine_times: int = 3

    @cached_property
    def t(self):
        """The coarse nodes on [0,2T]: 2N+2 of them."""
        return np.linspace(0, 2*self.T, 2*self.N + 2)

    @property
    def Δt(self):
        """The time step Δt."""
        return self.t[1] - self.t[0]

    def field(self, coefs):
        """The callable  s ↦ Σ_j coefs_j φ̂_j(s).

        A P1 function is its own piecewise linear interpolant, so this is just
        `np.interp` through the nodal values — exact, vectorised, and defined
        for scalars, which the time-stepping loops need."""
        y = np.zeros(len(self.t))
        y[1:len(coefs) + 1] = coefs
        return lambda s: np.interp(s, self.t, y, left=0.0, right=0.0)

    def φ̂(self, j):
        """The single hat function φ̂_j, as a callable."""
        return self.field(np.eye(2*self.N)[j])

    def Φ̂(self, s):
        """The matrix Φ̂_{js} = φ̂_j(s) of the first N hats, from skfem's
        `probes`: the whole basis evaluated in one shot."""
        basis, dofs, _, _ = _fem(self.N, self.T)
        return basis.probes(np.atleast_2d(s)).tocsr()[:, dofs].toarray().T

    @cached_property
    def t_refined(self):
        """The refined mesh on [0,2T] used for quadrature."""
        return np.linspace(0, 2*self.T,
                           2**self.refine_times*(2*self.N + 1) + 1)

    @property
    def refine_offset(self):
        """Refined steps per coarse step Δt."""
        return 2**self.refine_times

    @cached_property
    def refine_mask(self):
        """Which entries of `t_refined` are coarse nodes."""
        out = np.full(self.t_refined.shape, False)
        out[::self.refine_offset] = True
        return out

    @property
    def t_ixs_0_T(self):
        """Indices of t lying in [0,T]."""
        return np.arange(0, self.N + 1)

    @property
    def t_ixs_T_2T(self):
        """Indices of t lying in [T,2T]."""
        return np.arange(self.N + 1, 2*self.N + 2)

    @property
    def φ̂_ixs_0_T(self):
        """Indices of the hat functions centred in [0,T]."""
        return np.arange(0, self.N)

    @property
    def φ̂_ixs_T_2T(self):
        """Indices of the hat functions centred in [T,2T]."""
        return np.arange(self.N, 2*self.N)

    @property
    def φ̂_ixs(self):
        """Indices of all 2N hat functions."""
        return np.arange(0, 2*self.N)

    def t_node(self, i=None):
        """The centre t_{i+1} of φ̂_i."""
        return self.t[(self.φ̂_ixs if i is None else i) + 1]

    @property
    def refined_ix_T(self):
        """Where T sits inside `t_refined`."""
        return self.N*self.refine_offset + self.refine_offset//2


class InvProb(NamedTuple):
    """A discretisation, a wave speed, and the parameters behind the discrete
    truncated map Λ_L^ω: the truncation L, the impedance ω, and the spatial
    mesh N_x of (𝒮_ω) on Ω (h = 1/N_x).

    `data`, if given, is the spectral data (λ_0, λ̇_0) itself, e.g. with noise;
    the inverse map then uses it and never simulates it from c.  If None, the
    data are simulated from c by (𝒮_ω)."""
    discr: Discr = Discr()
    c: Callable = lambda x: np.ones_like(x)
    L: int = 150
    ω: float = 1e-4
    N_x: int = 600
    data: tuple = None

    @property
    def spectral_data(self):
        """(λ_0, λ̇_0): the given `data`, or the data simulated from c."""
        if self.data is not None:
            return self.data
        return _simulate_data(self.c, self.L, self.ω, self.N_x)

    @property
    def C_max(self):
        """max_Ω̄ c, sampled on 1001 points.  Only sets the resolution Δx of the
        direct solver `fwd.solve_wave`; it is not the C² bound C_max of the
        paper."""
        return float(np.max(self.c(np.linspace(0, 1, 1001))))

    @property
    def τ_1(self):
        """τ̂(1), the travel time across Ω estimated from the spectrum λ_n(0)
        alone (Weyl asymptotics); see `fwd.travel_time_from_spectrum`."""
        return fwd.travel_time_from_spectrum(self.spectral_data[0])


@lru_cache(maxsize=32)
def _simulate_data(c, L, ω, N_x):
    return fwd.spectral_data(c, L, ω, N_x)


def B1(t, T):
    """B applied to the constant 1.  With B[f](t) = −1_{(0,T)}(t)∫_t^T f,

        B1(t) = −1_{(0,T)}(t)·(T − t) = 1_{(0,T)}(t)·(t − T) ≤ 0.

    Used to assemble b; the sign is what makes the controls non-positive."""
    return np.where((0 <= t) & (t <= T), np.asarray(t, float) - T, 0.0)


def V_exact(c, n=1000):
    """The true  V(r) = ∫_{Ω(r)} c⁻² dx  on [0, τ(1)], for scoring.

    Built from τ(x) = ∫_0^x dy/c(y) and its inverse χ, both by interpolation of
    the same sampled curve.  Returns (r, V)."""
    x = np.linspace(0, 1, n)
    τ = cumulative_trapezoid(1/c(x), x, initial=0)
    V_of_x = cumulative_trapezoid(1/c(x)**2, x, initial=0)

    r = np.linspace(0, τ[-1], n)
    return r, np.interp(np.interp(r, τ, x), x, V_of_x)      # V(χ(r))


# =============================================================================
# The Neumann-to-Dirichlet map
# =============================================================================

def u(ip):
    """The field u^f of (𝒲_f) with f = φ̂_0, as (u, t, x).

    This is the direct route to Λ: its first column, u^f|_{x=0}, is the same
    trace `Λ_trace` produces from the spectra.  The two share no discretisation,
    which is what makes comparing them a check rather than a tautology."""
    t = ip.discr.t_refined
    x = fwd.x_grid(t, ip.C_max)
    return fwd.solve_wave(t, x, ip.c, ip.discr.φ̂(0)), t, x


def Λ_trace(ip):
    """The trace Λ_L^ω[φ̂_0] on the refined mesh, as (Λφ̂_0, t).

    The spectral route: the series of Proposition 1 truncated at L modes.  For
    the independent route take `u(ip)[0][:, 0]`."""
    t = ip.discr.t_refined
    return fwd.Λ_map(t, c=ip.c, f=ip.discr.φ̂(0), L=ip.L, ω=ip.ω, N_x=ip.N_x), t


def Λ(ip):
    """The matrix of Λ in the hat basis, shape (len(t_refined), N).

    Λ is a convolution, so column n is column 0 shifted by n·Δt: one trace
    fills the whole matrix.  K is built from this, always by the spectral
    route — the spectra are the data of the inverse problem, and a solve of
    (𝒲_f) would need the c we are trying to recover."""
    Λφ̂_0, t = Λ_trace(ip)
    shift = np.arange(len(t))[:, None] - ip.discr.refine_offset*np.arange(ip.discr.N)
    return np.where(shift >= 0, Λφ̂_0[np.clip(shift, 0, None)], 0.0)


# =============================================================================
# The Blagoveščenskiĭ operators
# =============================================================================

def _J_sampled(F, discr):
    """J[f](t) = ½·1_{(0,T)}(t)∫_t^{2T−t} f  for the columns of F sampled on
    `t_refined`; trapezoidal.  `t_refined` is symmetric about T, so 2T − t_i is
    the node t_{n−1−i}."""
    t, ix_T = discr.t_refined, discr.refined_ix_T
    G = cumulative_trapezoid(F, t, axis=0, initial=0)
    out = np.zeros_like(F)
    i = np.arange(ix_T + 1)
    out[i] = 0.5*(G[len(t) - 1 - i] - G[i])
    return out


def K(ip):
    """K_{jk} = (φ̂_j, Kφ̂_k)_{L²(0,2T)}  with  K = RΛRJ − JΛ.

    Every operator acts on functions sampled on `t_refined`, never on hat
    coefficients: J[φ̂_k] is not P1 (it is quadratic and cut at T, which lies
    inside an element), and R does not map hats of [0,T] to hats of [0,T].
    R is the flip of the symmetric mesh, Λ the convolution with the kernel of
    Λ_L^ω, J trapezoidal; the final pairing is trapezoidal on [0,T], where Kφ̂_k
    is supported.  The quadrature error is O(Δs²), Δs the refined step.

    K is symmetrised before it is returned; the antisymmetric part is
    quadrature error only."""
    d = ip.discr
    t, ix_T, Δs = d.t_refined, d.refined_ix_T, d.t_refined[1] - d.t_refined[0]
    k_Λ = fwd.kernel_from_data(t, *ip.spectral_data)
    Λ_apply = lambda F: fftconvolve(k_Λ[:, None], F, axes=0)[:len(t)] * Δs

    Φ = d.Φ̂(t).T                                         # (n_t, N): φ̂_k(t_i)
    RΛRJφ̂ = Λ_apply(_J_sampled(Φ, d)[::-1])[::-1]
    JΛφ̂ = _J_sampled(Λ_apply(Φ), d)
    Kφ̂ = (RΛRJφ̂ - JΛφ̂)[:ix_T + 1]                       # supported in [0,T]

    w = np.full(ix_T + 1, Δs)
    w[[0, -1]] *= 0.5
    K_h = Φ[:ix_T + 1].T @ (Kφ̂ * w[:, None])
    return 0.5*(K_h + K_h.T)


def b(discr):
    """b_j = (φ̂_j, B1)_{L²(0,2T)}  with  B1(t) = 1_{(0,T)}(t)·(t − T) ≤ 0.

    On each half of a hat's support the integrand φ̂_j(t)(t − T) is quadratic,
    so the two-point Gauss rule is exact.  T falls strictly inside the element
    (t_N, t_{N+1}), so the sub-intervals are clipped at T rather than special
    cased: the hats past T contribute nothing and the straddling one is cut."""
    N, T, t, Δt = discr.N, discr.T, discr.t, discr.Δt
    ξ, w = np.polynomial.legendre.leggauss(2)

    lo = np.stack([t[:N], t[1:N+1]], 1)                        # (N,2) ends
    hi = np.maximum(np.minimum(np.stack([t[1:N+1], t[2:N+2]], 1), T), lo)
    s = (lo + hi)[..., None]/2 + (hi - lo)[..., None]/2 * ξ    # (N,2,2) nodes
    φ̂ = np.stack([(s[:, 0] - t[:N, None])/Δt,                  # rising half
                   (t[2:N+2, None] - s[:, 1])/Δt], 1)          # falling half
    return (w * φ̂ * B1(s, T) * ((hi - lo)/2)[..., None]).sum((1, 2))


def M(ip):
    """M_{jk} = (φ̂_j, φ̂_k)_{L²(0,2T)}: the identity I of  P_r K P_r + αI + βS.

    Shared out of the cache, so treat it as read-only."""
    return _fem(ip.discr.N, ip.discr.T)[2]


def S(ip):
    """S_{jk} = (∂ₜφ̂_j, ∂ₜφ̂_k)_{L²(0,2T)}: the stiffness operator S.

    Shared out of the cache, so treat it as read-only."""
    return _fem(ip.discr.N, ip.discr.T)[3]


# =============================================================================
# The control problems and the volume
# =============================================================================

def Q(α, β, ip):
    """The pair (Q, b) of the control problem, Q = K + αM + βS.

    The single place where K, M, S and b are built, so each is assembled once
    no matter which entry point asked for it."""
    return K(ip) + α*M(ip) + β*S(ip), b(ip.discr)


def _solve_over_r(Q_h, b_h):
    """f_αβr for every r, solving Q f = P_r b.  P_r keeps the trailing r hats,
    so the r-th problem is the trailing r × r block.  Returns an (N, N) array,
    row r−1 being f_αβr — non-positive, since b ≤ 0.

    Reversing the index order turns every trailing block into a leading one,
    and the Cholesky factor of a leading principal block is the leading block
    of the full factor.  One factorisation, O(N³), then two triangular solves
    of size r per r; the naive loop was O(N⁴).  Falls back to dense solves if
    Q is not numerically positive definite."""
    N = len(b_h)
    Q̃, b̃ = Q_h[::-1, ::-1], b_h[::-1]
    F = np.zeros((N, N))
    try:
        L_fac = cho_factor(0.5*(Q̃ + Q̃.T), lower=True)[0]
    except np.linalg.LinAlgError:
        for r in range(1, N + 1):
            F[r-1, N-r:] = np.linalg.solve(Q_h[N-r:, N-r:], b_h[N-r:])
        return F
    for r in range(1, N + 1):
        L_r = L_fac[:r, :r]
        y = solve_triangular(L_r, b̃[:r], lower=True)
        F[r-1, N-r:] = solve_triangular(L_r.T, y, lower=False)[::-1]
    return F


def f_αβr(α, β, ip):
    """The minimisers of Proposition 2, one row per r on the grid of hat
    centres:  f_αβr = (P_r K P_r + αI + βS)⁻¹ P_r B1.

    Each row is non-positive, peaking near t = T − r: in the homogeneous case
    the exact control is −δ_{T−r}."""
    return _solve_over_r(*Q(α, β, ip))


def V(α, β, ip):
    """The travel-time volume  V(r) = (f_αβr, B1) = (u^{f_αβr}(T), 1)_{L²_c}
    ≈ ∫_{Ω(r)} c⁻² dx.  Positive: both factors are non-positive.

    r runs over the hat centres t_1, …, t_N of [0,T], truncated to r ≤ τ̂(1):
    past τ(1) the second Blagoveščenskiĭ identity fails.  Returns (V, r)."""
    Q_h, b_h = Q(α, β, ip)
    r = ip.discr.t_node(np.arange(ip.discr.N))
    keep = r <= ip.τ_1
    if not keep.all():
        warnings.warn(f"T = {ip.discr.T:.5f} > τ̂(1) = {ip.τ_1:.5f}: the "
                      f"{(~keep).sum()} controls with r > τ̂(1) are dropped.",
                      RuntimeWarning, stacklevel=2)
    return (_solve_over_r(Q_h, b_h) @ b_h)[keep], r[keep]


# ASCII aliases for the public API
Lambda, Lambda_trace, f_abr = Λ, Λ_trace, f_αβr
