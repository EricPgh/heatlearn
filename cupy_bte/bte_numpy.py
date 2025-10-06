"""
CuPy-based 1D phonon BTE solver with RTA and first-order upwind advection.

Model summary
-------------
We evolve the deviation-from-equilibrium distribution `g` for two phonon branches
(i = 0, 1) and two propagation directions s in {+1 (right), -1 (left)} over a 1D
spatial grid x in [0, L]. The linearized, grey-ish form is:

    ?g_{i,s}/?t + s * v_i * ?g_{i,s}/?x = - g_{i,s}/?_i + S_{i,s}(x,t)

Here we take S = 0 and drive the system via *inflow boundary values* that
represent a small temperature offset at the boundaries, e.g. ?T_L at x=0 for
right-going phonons and ?T_R at x=L for left-going phonons. The state variable
`g` is measured in energy density units so that a local temperature perturbation
is ?(x,t) = (?_i,s g_{i,s}) / (?_i C_i), with C_i branch heat capacities.

Numerics
--------
- Space: uniform grid, first-order upwind differencing (directional).
- Time: explicit forward Euler. Stable if CFL <= 1 and dt << min(?_i) for RTA.
- Boundaries: inflow Dirichlet for g: at left boundary for s=+1, at right
  boundary for s=-1. Outflow boundaries are naturally upwinded.

This is a minimal, hackable skeleton tailored to your description. It runs fully
on the GPU via CuPy, with host transfers only for diagnostics/plotting.

Dependencies
------------
- cupy (e.g. `pip install cupy-cuda12x`) matching your CUDA version
- matplotlib (optional, for plotting diagnostics)

Tip: start with small Nx (<= 2048) and tighten dt by reducing `cfl` or `dt_cap`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
try:
    import cupy as cp
except Exception as e:
    raise SystemExit("CuPy is required. Install a CUDA-matched build, e.g. 'pip install cupy-cuda12x'.")


@dataclass
class BTEParams:
    L: float = 1e-3                 # domain length [m]
    Nx: int = 2048                  # grid points
    v: Tuple[float, float] = (3000.0, 1500.0)  # group speeds for branches [m/s]
    tau: Tuple[float, float] = (2e-9, 5e-9)    # relaxation times [s]
    C: Tuple[float, float] = (1.0e6, 1.0e6)    # branch heat capacities [J/(m^3 K)]
    dTL: float = 1.0                 # ?T at x=0 for right-going phonons [K]
    dTR: float = -1.0                # ?T at x=L for left-going phonons [K]
    cfl: float = 0.9                 # CFL number (<= 1 for upwind)
    dt_cap: Optional[float] = None   # optional hard cap on dt [s]
    t_final: float = 2e-6            # final time [s]
    save_every: int = 200            # steps between diagnostics
    seed: Optional[int] = 1234       # for optional perturbations


class BTESolver:
    def __init__(self, p: BTEParams):
        self.p = p
        self.x = cp.linspace(0.0, p.L, p.Nx)
        self.dx = p.L / (p.Nx - 1)
        self.inventory = cp.array([0.]*100+[1e8]*1848+[0.]*100).reshape((1,2048))
        print(self.inventory)

        # Branch params as device arrays
        self.v = cp.asarray(p.v, dtype=cp.float64)         # (2,)
        self.tau = cp.asarray(p.tau, dtype=cp.float64)     # (2,)
        self.C = cp.asarray(p.C, dtype=cp.float64)         # (2,)
        self.Ctot = float(cp.sum(self.C).get())

        # State: g[i, s, x] with i in {0,1}, s in {0: +1, 1: -1}
        self.g = cp.zeros((2, 2, p.Nx), dtype=cp.float64)

        # Precompute stable dt from CFL and RTA
        vmax = float(cp.max(cp.abs(self.v)).get())
        dt_cfl = p.cfl * self.dx / max(vmax, 1e-30)
        dt_rta = float(cp.min(self.tau).get()) / 10.0
        dt = min(dt_cfl, dt_rta)
        if p.dt_cap is not None:
            dt = min(dt, p.dt_cap)
        self.dt = dt

        # Boundary inflow values for g = C_i * ?T_inflow / 2 (split across directions)
        # Right-going (+): inflow at left boundary uses ?T_L
        # Left -going (-): inflow at right boundary uses ?T_R
        self.g_in_L = (self.C * p.dTL / 2.0)[:, None]  # shape (2,1)
        self.g_in_R = (self.C * p.dTR / 2.0)[:, None]  # shape (2,1)

    def step(self):
        p = self.p
        dt = self.dt
        dx = self.dx
        v = self.v  # (2,)
        tau = self.tau  # (2,)
        g = self.g  # (2,2,Nx)

        # Advection: first-order upwind, handled per direction
        # s_idx=0 corresponds to s=+1, s_idx=1 corresponds to s=-1
        # For s=+1: ?g/?x ? (g[x] - g[x-1]) / dx, with inflow at x=0 set to g_in_L
        g_plus = g[:, 0, :]  # (2,Nx)
        g_plus_up = cp.roll(g_plus, 1, axis=-1)
        # set upwind neighbor at the left boundary to inflow value
        g_plus_up[:, 0] = self.g_in_L[:, 0]
        dgdx_plus = (g_plus - g_plus_up) / dx

        # For s=-1: ?g/?x ? (g[x+1] - g[x]) / dx, with inflow at x=L set to g_in_R
        g_minus = g[:, 1, :]
        g_minus_dn = cp.roll(g_minus, -1, axis=-1)
        g_minus_dn[:, -1] = self.g_in_R[:, 0]
        dgdx_minus = (g_minus_dn - g_minus) / dx

        # Advection update: g_t = - s*v * dgdx - g/tau
        # Broadcast v and tau to (2,Nx)
        v2 = v[:, None]
        tau2 = tau[:, None]

        rhs_plus = - (+1.0) * (v2 * dgdx_plus) - (g_plus / tau2)
        rhs_minus = - (-1.0) * (v2 * dgdx_minus) - (g_minus / tau2)

        rxn = (g[:,0,:]+g[:,1,:])*self.inventory*1e0
        sink = rxn+100.
        #print(rhs_plus.shape,rhs_minus.shape)
        #print(g.shape,g[:,0,:].shape)
        #print(sink.shape,self.inventory.shape)
        #exit()
        self.inventory -= dt*(rxn[0,:]+rxn[1,:])
        self.inventory = cp.maximum(self.inventory,cp.zeros_like(self.inventory))
        #sink = cp.array([[min(0.,1-5*(i-1024)**2)]*2 for i in range(self.p.Nx)]).T
        g[:, 0, :] = g_plus + dt * (rhs_plus - sink)
        g[:, 1, :] = g_minus + dt * (rhs_minus - sink)

        # Enforce inflow Dirichlet at boundaries explicitly after the update
        g[:, 0, 0] = self.g_in_L[:, 0]  # right-going at left boundary
        g[:, 1, -1] = self.g_in_R[:, 0] # left-going at right boundary

    def temperature(self) -> cp.ndarray:
        # ? = (sum over i,s g_{i,s}) / ?_i C_i
        return cp.sum(self.g, axis=(0,1)) / self.Ctot  # (Nx,)

    def run(self, progress: bool = True):
        t = 0.0
        nsteps = int(np.ceil(self.p.t_final / self.dt))
        ts = []
        Tsnaps = []
        every = max(1, self.p.save_every)

        for n in range(nsteps):
            self.step()
            t += self.dt
            if n % every == 0 or n == nsteps - 1:
                Ts = self.temperature().get()  # host copy for diagnostics
                ts.append(t)
                #Tsnaps.append(Ts)
                Tsnaps.append(self.inventory.get())
                if progress:
                    print(f"t = {t:.3e} s  (dt={self.dt:.3e}, step {n+1}/{nsteps})  Tmean={Ts.mean():+.3e} K")
        return np.array(ts), np.array(Tsnaps)#np.stack(Tsnaps, axis=0)


def demo():
    import pickle
    p = BTEParams(
        L=1e-3,
        Nx=2048,
        v=(3000.0, 1500.0),
        tau=(8e-3, 8e-3),
        C=(1.0e7, 1.0e6),
        dTL=1e0,      # +1 K at left
        dTR=3e0,     # -1 K at right
        cfl=0.9,
        t_final=2e-6,
        save_every=1e0,
    )

    solver = BTESolver(p)
    ts, Tsnaps = solver.run(progress=True)
    with open('bte_1d.pkl','wb') as file:
        pickle.dump([ts,Tsnaps],file)

    if True:#try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams.update({"figure.figsize": (7, 4)})
        x = cp.asnumpy(solver.x)
        for Ts in Tsnaps:#[np.array([1,6,-1])]:
            plt.plot(x * 1e6, Ts.T)#, label="?(x, t_final)")
        plt.xlabel("x [m]")
        #plt.ylabel("temperature perturbation ? [K]")
        plt.title("1D Inventory from Upwind BTE")
        #plt.legend()
        plt.tight_layout()
        plt.xlim(0,1000)
        plt.savefig('bte_temps.png')
        plt.show()
    else:#except Exception as e:
        print("Plotting skipped (matplotlib not available)")


if __name__ == "__main__":
    demo()

