import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constants (SI)
# -----------------------------
hbar = 1.054_571_817e-34  # J*s
uGal = 1e-8               # 1 μGal = 1e-8 m/s^2

# -----------------------------
# Fig. 2 parameters (caption)
# -----------------------------
omega  = 2*np.pi*10e3      # rad/s, ω/2π = 10 kHz
B0     = 50e3              # T/m, 50 kT/m
r      = 100e-9            # m, radius
k_eff  = 1.6e7             # 1/m, atom interferometer effective wavevector
gamma_e = 2*np.pi*28e9     # rad/s/T, since γe/2π = 28 GHz/T :contentReference[oaicite:1]{index=1}

# Diamond density (not stated in caption; use standard value; small effect)
rho = 3500.0               # kg/m^3

# -----------------------------
# Derived quantities
# -----------------------------
M  = (4/3)*np.pi*r**3*rho                # kg
y0 = np.sqrt(hbar/(2*M*omega))           # m, zero-point motion scale
eta = (gamma_e*B0*y0)/omega              # dimensionless

tau = 2*np.pi/omega
overhead = 7*tau/2                       # Δt_min in their protocol :contentReference[oaicite:2]{index=2}

# -----------------------------
# Sensitivity models
# -----------------------------
def S_ours(dt):
    """
    Uses the paper's closed-form sensitivity:
    S = ω y0 sqrt(Δt) / |16π η M y0^2/ħ + 2 η ω T^2|
    with Δt = 7τ/2 + 2T. :contentReference[oaicite:3]{index=3}
    """
    dt = np.asarray(dt, dtype=float)
    T = (dt - overhead)/2
    S = np.full_like(dt, np.nan)

    mask = T > 0
    denom = np.abs(16*np.pi*eta*M*(y0**2)/hbar + 2*eta*omega*(T[mask]**2))
    S[mask] = (omega*y0*np.sqrt(dt[mask]))/denom
    return S

def S_linear(dt):
    """
    Linear accumulation: ΔΦ2 = 4 η λg T for duration T. :contentReference[oaicite:4]{index=4}
    In SI, λg = M g y0 / ħ, so ∂ΔΦ2/∂g = 4 η (M y0/ħ) T.
    Identify Δt = T.
    """
    dt = np.asarray(dt, dtype=float)
    T = dt
    deriv = 4*eta*(M*y0/hbar)*T
    return np.sqrt(dt)/np.abs(deriv)

def S_atom(dt):
    """
    Atom: ΔΦ1 = k_eff g T^2, and total time Δt = 2T. :contentReference[oaicite:5]{index=5}
    => T = Δt/2, ∂ΔΦ1/∂g = k_eff T^2
    """
    dt = np.asarray(dt, dtype=float)
    T = dt/2
    deriv = k_eff*T**2
    return np.sqrt(dt)/np.abs(deriv)

# -----------------------------
# Sweep and plot (paper-like window)
# -----------------------------
dt = np.logspace(np.log10(overhead*1.01), np.log10(20), 900)

S1 = S_ours(dt)/uGal
S2 = S_linear(dt)/uGal
S3 = S_atom(dt)/uGal

fig, ax = plt.subplots(figsize=(7.6, 4.9))
ax.loglog(dt, S1, lw=3, color="m", label="Our protocol (square-accelerated)")
ax.loglog(dt, S2, lw=3, ls="--", color="tab:orange", label="Linear accumulation (Ref. 55)")
ax.loglog(dt, S3, lw=3, ls="-.", color="tab:blue", label="Single atom gravimeter")

# Match the paper's visible range (this is the big difference vs your plot)
ax.set_xlim(4e-4, 2e1)
ax.set_ylim(1e0, 2e6)

ax.set_xlabel(r"Sensing time $\Delta t$ (s)")
ax.set_ylabel(r"$S$ ($\mu$Gal$/\sqrt{\mathrm{Hz}}$)")

# Add the 20 μGal/√Hz guide like the paper text discussion
target = 20.0
ax.axhline(target, color="k", lw=1.5, ls=":")

def closest_dt(dt_arr, S_arr, target_val):
    idx = np.nanargmin(np.abs(S_arr - target_val))
    return dt_arr[idx], S_arr[idx]

dt_ours, _ = closest_dt(dt, S1, target)
dt_lin,  _ = closest_dt(dt, S2, target)

ax.plot([dt_ours, dt_lin], [target, target], "ko", ms=5)
ax.text(dt_lin*1.05, target*1.08, r"$20\ \mu$Gal$/\sqrt{\mathrm{Hz}}$", fontsize=12)

ax.legend(loc="upper right", frameon=True)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.grid(False)

plt.tight_layout()
# fig.savefig("./figure2_reproduction.png", dpi=300, bbox_inches="tight")
fig.savefig("./figure2_reproduction.pdf", bbox_inches="tight")
# plt.show()

print(f"Δt (ours) for ~20 μGal/√Hz:  {dt_ours:.4g} s")
print(f"Δt (linear) for ~20 μGal/√Hz: {dt_lin:.4g} s")
