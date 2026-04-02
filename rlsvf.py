import numpy as np
import matplotlib.pyplot as plt

#=============
# Hyperparameters
#=============

N_ORDER      = 9
N_SAMPLES    = 3000
N_TRIALS     = 30
SNR_DB       = 25.0
RHO_FIXED    = 0.98
RHO_MIN      = 0.85
N_MAX        = 200.0
L_WINDOW     = 5
LAMBDA_S     = 0.99
P0_SCALE     = 100.0
OUTLIER_PROB = 0.01
OUTLIER_VAR  = 1e4 / 12.0

W_TRUE_BASE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
STEP_TIMES  = [500, 1000, 1500, 2000, 2500]
STEP_VALUES = [0.5,  0.1,  0.5,  0.1,  0.5]
NOISE_STD   = np.sqrt(1.0 / (10 ** (SNR_DB / 10.0)))

#=============
# Storage arrays
#=============

nee_rls_clean   = np.zeros((N_TRIALS, N_SAMPLES))
nee_rlsvf_clean = np.zeros((N_TRIALS, N_SAMPLES))
rho_clean       = np.zeros((N_TRIALS, N_SAMPLES))
w5_rls_clean    = np.zeros((N_TRIALS, N_SAMPLES))
w5_rlsvf_clean  = np.zeros((N_TRIALS, N_SAMPLES))

nee_rls_noisy   = np.zeros((N_TRIALS, N_SAMPLES))
nee_rlsvf_noisy = np.zeros((N_TRIALS, N_SAMPLES))
rho_noisy       = np.zeros((N_TRIALS, N_SAMPLES))
w5_rls_noisy    = np.zeros((N_TRIALS, N_SAMPLES))
w5_rlsvf_noisy  = np.zeros((N_TRIALS, N_SAMPLES))

w5_true_stored = None

#=============
# Monte Carlo trials
#=============

for use_outliers in [False, True]:


    for trial in range(N_TRIALS):

        rng = np.random.default_rng(trial * 42)

        #=============
        # Data generation
        #=============

        x = rng.standard_normal(N_SAMPLES + N_ORDER)

        W_true = np.tile(W_TRUE_BASE, (N_SAMPLES, 1))
        for t, v in zip(STEP_TIMES, STEP_VALUES):
            W_true[t:, 4] = v

        X_mat = np.zeros((N_SAMPLES, N_ORDER))
        for k in range(N_SAMPLES):
            X_mat[k] = x[k + N_ORDER - 1: k - 1: -1] if k > 0 else x[N_ORDER - 1::-1]

        d_clean = np.zeros(N_SAMPLES)
        for k in range(N_SAMPLES):
            d_clean[k] = X_mat[k] @ W_true[k]

        noise = rng.standard_normal(N_SAMPLES) * NOISE_STD
        if use_outliers:
            alpha = (rng.uniform(size=N_SAMPLES) < OUTLIER_PROB).astype(float)
            A = rng.standard_normal(N_SAMPLES) * np.sqrt(OUTLIER_VAR)
            noise = noise + alpha * A

        d = d_clean + noise

        if w5_true_stored is None:
            w5_true_stored = W_true[:, 4].copy()

        #=============
        # Initialisation
        #=============

        W_rls = np.zeros(N_ORDER)
        P_rls = P0_SCALE * np.eye(N_ORDER)

        W_rlsvf = np.zeros(N_ORDER)
        P_rlsvf = P0_SCALE * np.eye(N_ORDER)
        s2      = 1.0
        err_buf = np.zeros(L_WINDOW)

        #=============
        # Main time loop
        #=============

        for k in range(N_SAMPLES):

            x_k    = X_mat[k]
            d_k    = d[k]
            w_true = W_true[k]

            #=============
            # Standard RLS update
            #=============

            e_rls     = d_k - x_k @ W_rls
            denom_rls = RHO_FIXED + x_k @ P_rls @ x_k
            K_rls     = (P_rls @ x_k) / denom_rls
            W_rls     = W_rls + K_rls * e_rls
            P_rls     = (1.0 / RHO_FIXED) * (P_rls - np.outer(K_rls, x_k) @ P_rls)

            #=============
            # Variable forgetting factor
            #=============

            e_pre       = d_k - x_k @ W_rlsvf
            err_buf     = np.roll(err_buf, -1)
            err_buf[-1] = e_pre
            E_k         = np.mean(err_buf ** 2)
            s2          = LAMBDA_S * s2 + (1.0 - LAMBDA_S) * e_pre ** 2
            Q_k         = max(E_k / max(s2, 1e-12), 1.0)
            N_k         = N_MAX / Q_k
            rho_k       = max(1.0 - 1.0 / N_k, RHO_MIN)

            #=============
            # RLSVF update
            #=============

            e_rlsvf     = d_k - x_k @ W_rlsvf
            denom_rlsvf = rho_k + x_k @ P_rlsvf @ x_k
            K_rlsvf     = (P_rlsvf @ x_k) / denom_rlsvf
            W_rlsvf     = W_rlsvf + K_rlsvf * e_rlsvf
            P_rlsvf     = (1.0 / rho_k) * (P_rlsvf - np.outer(K_rlsvf, x_k) @ P_rlsvf)

            #=============
            # NEE
            #=============

            norm_true   = max(np.linalg.norm(w_true) ** 2, 1e-30)
            nee_rls_k   = 10.0 * np.log10(np.linalg.norm(W_rls   - w_true) ** 2 / norm_true)
            nee_rlsvf_k = 10.0 * np.log10(np.linalg.norm(W_rlsvf - w_true) ** 2 / norm_true)

            if use_outliers:
                nee_rls_noisy[trial, k]   = nee_rls_k
                nee_rlsvf_noisy[trial, k] = nee_rlsvf_k
                rho_noisy[trial, k]       = rho_k
                w5_rls_noisy[trial, k]    = W_rls[4]
                w5_rlsvf_noisy[trial, k]  = W_rlsvf[4]
            else:
                nee_rls_clean[trial, k]   = nee_rls_k
                nee_rlsvf_clean[trial, k] = nee_rlsvf_k
                rho_clean[trial, k]       = rho_k
                w5_rls_clean[trial, k]    = W_rls[4]
                w5_rlsvf_clean[trial, k]  = W_rlsvf[4]

#=============
# Average across trials
#=============

k_axis = np.arange(N_SAMPLES)

mean_nee_rls_clean   = np.mean(nee_rls_clean,   axis=0)
mean_nee_rlsvf_clean = np.mean(nee_rlsvf_clean, axis=0)
mean_rho_clean       = np.mean(rho_clean,        axis=0)
mean_w5_rls_clean    = np.mean(w5_rls_clean,     axis=0)
mean_w5_rlsvf_clean  = np.mean(w5_rlsvf_clean,   axis=0)

mean_nee_rls_noisy   = np.mean(nee_rls_noisy,   axis=0)
mean_nee_rlsvf_noisy = np.mean(nee_rlsvf_noisy, axis=0)
mean_rho_noisy       = np.mean(rho_noisy,        axis=0)
mean_w5_rls_noisy    = np.mean(w5_rls_noisy,     axis=0)
mean_w5_rlsvf_noisy  = np.mean(w5_rlsvf_noisy,   axis=0)

#=============
# Plotting
#=============

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Experiment 4.1 — Kovacevic et al. (2016)\nRLS vs RLSVF: Time-Varying Parameter Tracking")

axes[0, 0].plot(k_axis, w5_true_stored, '--', label="True w5")
axes[0, 0].plot(k_axis, mean_w5_rls_clean,   label="RLS (fixed FF)")
axes[0, 0].plot(k_axis, mean_w5_rlsvf_clean, label="RLSVF")
axes[0, 0].set_title("w5 Tracking — Gaussian Noise")
axes[0, 0].set_ylabel("Parameter value")
axes[0, 0].set_ylim(-0.2, 0.9)
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(k_axis, w5_true_stored, '--', label="True w5")
axes[0, 1].plot(k_axis, mean_w5_rls_noisy,   label="RLS (fixed FF)")
axes[0, 1].plot(k_axis, mean_w5_rlsvf_noisy, label="RLSVF")
axes[0, 1].set_title("w5 Tracking — Gaussian + Impulsive Noise")
axes[0, 1].set_ylabel("Parameter value")
axes[0, 1].set_ylim(-0.2, 0.9)
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(k_axis, mean_rho_clean, label="rho(k) — RLSVF")
axes[1, 0].axhline(RHO_FIXED, linestyle='--', label=f"Fixed rho = {RHO_FIXED}")
axes[1, 0].set_title("Variable Forgetting Factor — Gaussian Noise")
axes[1, 0].set_ylabel("rho(k)")
axes[1, 0].set_ylim(0.80, 1.01)
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(k_axis, mean_rho_noisy, label="rho(k) — RLSVF")
axes[1, 1].axhline(RHO_FIXED, linestyle='--', label=f"Fixed rho = {RHO_FIXED}")
axes[1, 1].set_title("Variable Forgetting Factor — Gaussian + Impulsive Noise")
axes[1, 1].set_ylabel("rho(k)")
axes[1, 1].set_ylim(0.80, 1.01)
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[2, 0].plot(k_axis, mean_nee_rls_clean,   label="RLS (fixed FF)")
axes[2, 0].plot(k_axis, mean_nee_rlsvf_clean, label="RLSVF")
axes[2, 0].set_title(f"Ensemble-Averaged NEE ({N_TRIALS} runs) — Gaussian Noise")
axes[2, 0].set_xlabel("Iteration k")
axes[2, 0].set_ylabel("NEE (dB)")
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(k_axis, mean_nee_rls_noisy,   label="RLS (fixed FF)")
axes[2, 1].plot(k_axis, mean_nee_rlsvf_noisy, label="RLSVF")
axes[2, 1].set_title(f"Ensemble-Averaged NEE ({N_TRIALS} runs) — Gaussian + Impulsive Noise")
axes[2, 1].set_xlabel("Iteration k")
axes[2, 1].set_ylabel("NEE (dB)")
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

#=============
# Summary
#=============

print(f"  [Clean]     RLS: {np.mean(mean_nee_rls_clean[-100:]):.2f} dB  |  RLSVF: {np.mean(mean_nee_rlsvf_clean[-100:]):.2f} dB")
print(f"  [Impulsive] RLS: {np.mean(mean_nee_rls_noisy[-100:]):.2f} dB  |  RLSVF: {np.mean(mean_nee_rlsvf_noisy[-100:]):.2f} dB")
