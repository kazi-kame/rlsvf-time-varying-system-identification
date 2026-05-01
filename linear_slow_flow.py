import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#=================================
# system parameters
#=================================
omega_n         = 2 * np.pi
zeta            = 0.05
x0, v0         = 1.0, 0.0
decay_rate_true = zeta * omega_n

#=================================
# RK4 simulation of: x'' + 2*zeta*w0*x' + w0^2*x = 0
#=================================
def ode_rhs(state, omega_n, zeta):
    x, v  = state
    x_dot = v
    v_dot = -(omega_n**2) * x - 2 * zeta * omega_n * v
    return np.array([x_dot, v_dot])

def rk4_step(state, dt, omega_n, zeta):
    k1 = ode_rhs(state,           omega_n, zeta)
    k2 = ode_rhs(state + dt/2*k1, omega_n, zeta)
    k3 = ode_rhs(state + dt/2*k2, omega_n, zeta)
    k4 = ode_rhs(state + dt*k3,   omega_n, zeta)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

dt    = 0.005
t_end = 20.0
t     = np.arange(0, t_end, dt)
N     = len(t)

states    = np.zeros((N, 2))
states[0] = [x0, v0]

for i in range(1, N):
    states[i] = rk4_step(states[i-1], dt, omega_n, zeta)

x = states[:, 0]
v = states[:, 1]

#=================================
# extract slow envelope via local maxima of x(t)
#=================================
peak_indices, _ = find_peaks(x, height=0)
t_peaks         = t[peak_indices]
A_peaks         = x[peak_indices]

#=================================
# log-linearise and fit with least squares
# MMS slow flow: A(t) = A0 * exp(-zeta*w0*t)
# log(A) = log(A0) - zeta*w0*t  =>  straight line in t
# LS recovers slope = -zeta*w0
#=================================
y   = np.log(A_peaks)
Phi = np.column_stack([np.ones_like(t_peaks), t_peaks])

coeffs, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
c0_fit, c1_fit  = coeffs

A0_identified         = np.exp(c0_fit)
decay_rate_identified = -c1_fit
zeta_identified       = decay_rate_identified / omega_n

print("\n--- Parameter Identification Results ---")
print(f"  True   decay rate (zeta*w0) : {decay_rate_true:.5f} rad/s")
print(f"  Fitted decay rate (zeta*w0) : {decay_rate_identified:.5f} rad/s")
print(f"  Error                       : {100*abs(decay_rate_true - decay_rate_identified)/decay_rate_true:.3f} %")
print(f"\n  True   zeta : {zeta:.4f}")
print(f"  Fitted zeta : {zeta_identified:.4f}")
print("----------------------------------------")

#=================================
# reconstruct slow envelope from identified parameters
#=================================
A_fit = A0_identified * np.exp(-decay_rate_identified * t_peaks)

#=================================
# plots
#=================================
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
fig.suptitle("Linear Oscillator - Slow-Flow Identification via Least Squares",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(t, x, color='steelblue', lw=0.8, alpha=0.8, label='x(t) [fast oscillation]')
ax.scatter(t_peaks, A_peaks, color='crimson', s=18, zorder=5, label='Peak samples A(t)')
ax.plot(t_peaks,  A_fit, color='darkorange', lw=2, linestyle='--',
        label=f'LS fit: A0*exp(-zeta*w0*t),  zeta={zeta_identified:.4f}')
ax.plot(t_peaks, -A_fit, color='darkorange', lw=2, linestyle='--', alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('x(t)')
ax.set_title('Time Series with Slow Envelope')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(t_peaks, np.log(A_peaks), color='crimson', s=18, zorder=5,
           label='log A (from peaks)')
ax.plot(t_peaks, c0_fit + c1_fit * t_peaks, color='darkorange', lw=2, linestyle='--',
        label=f'LS line: slope = -zeta*w0 = {c1_fit:.5f}')
ax.set_xlabel('Time [s]')
ax.set_ylabel('log A(t)')
ax.set_title('Log-Envelope vs Time (linearity confirms exponential decay)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
params      = ['zeta (damping ratio)', 'zeta*w0 (decay rate)']
true_vals   = [zeta,            decay_rate_true]
fitted_vals = [zeta_identified, decay_rate_identified]

x_pos = np.arange(len(params))
width = 0.35
bars1 = ax.bar(x_pos - width/2, true_vals,   width, label='True',      color='steelblue')
bars2 = ax.bar(x_pos + width/2, fitted_vals, width, label='Identified', color='darkorange')
ax.set_xticks(x_pos)
ax.set_xticklabels(params)
ax.set_ylabel('Value')
ax.set_title('True vs Identified Parameters')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()