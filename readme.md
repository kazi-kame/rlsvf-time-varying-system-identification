# Adaptive System Identification using RLS with Variable Forgetting Factor

Reproduction of the time-varying FIR system identification experiment from
**Kovačević et al., EURASIP Journal on Advances in Signal Processing (2016)**.

This project compares:

* Standard Recursive Least Squares (RLS)
* RLS with an **Extended-Prediction-Error based Variable Forgetting Factor (RLSVF)**

for tracking a **time-varying system** under:

* Gaussian noise
* Gaussian + impulsive noise

---

## Objective

To study how **adaptive memory** improves parameter tracking in non-stationary systems.

Key idea:

> A fixed-memory estimator cannot simultaneously achieve fast tracking and low steady-state variance.
> RLSVF adapts its memory in real time based on the prediction error.

---

## System Model

We identify a linear FIR system:

[
d(k) = \mathbf{w}^T(k)\mathbf{x}(k) + n(k)
]

where:

* ( \mathbf{x}(k) ) — regressor (input vector)
* ( \mathbf{w}(k) ) — true parameter vector
* ( n(k) ) — measurement noise

One coefficient of ( \mathbf{w}(k) ) changes in steps → **non-stationary plant**.

---

## Standard RLS

RLS minimises the exponentially weighted cost:

[
J(k) = \sum_{i=1}^{k} \rho^{k-i} e^2(i)
]

Update equations:

[
\mathbf{K}(k) = \frac{\mathbf{P}(k-1)\mathbf{x}(k)}{\rho + \mathbf{x}^T(k)\mathbf{P}(k-1)\mathbf{x}(k)}
]

[
\mathbf{w}(k) = \mathbf{w}(k-1) + \mathbf{K}(k)e(k)
]

[
\mathbf{P}(k) = \frac{1}{\rho}
\left[
\mathbf{P}(k-1) - \mathbf{K}(k)\mathbf{x}^T(k)\mathbf{P}(k-1)
\right]
]

where:

* ( \rho ) — forgetting factor
* ( \mathbf{P}(k) ) — covariance (inverse information matrix)

---

## Memory Interpretation

The forgetting factor corresponds to an **effective data window**:

[
N \approx \frac{1}{1 - \rho}
]

* Large ( \rho ) → long memory → low noise → slow tracking
* Small ( \rho ) → short memory → fast tracking → high variance

---

## RLS with Variable Forgetting Factor (RLSVF)

Instead of fixing ( \rho ), we estimate it from the data.

### Step 1 — Prediction error

[
e(k) = d(k) - \mathbf{x}^T(k)\mathbf{w}(k-1)
]

### Step 2 — Local error energy (EPE)

[
E(k) = \frac{1}{L}\sum_{i=k-L+1}^{k} e^2(i)
]

This measures **recent model mismatch**.

### Step 3 — Normalised non-stationarity measure

[
Q(k) = \frac{E(k)}{\hat{\sigma}^2(k)}
]

where ( \hat{\sigma}^2(k) ) is the global error variance.

Interpretation:

* (Q(k) \approx 1) → system is stationary
* (Q(k) > 1) → parameter change detected

### Step 4 — Adaptive memory length

[
N(k) = \frac{N_{\max}}{Q(k)}
]

### Step 5 — Forgetting factor

[
\rho(k) = 1 - \frac{1}{N(k)}
]

So the algorithm:

* shortens memory during transients
* lengthens memory in the steady state

---

## Performance Metric

We use the **Normalised Estimation Error (NEE)**:

[
\text{NEE}(k) =
10\log_{10}
\left(
\frac{|\mathbf{w}(k) - \mathbf{w}*{\text{true}}(k)|^2}
{|\mathbf{w}*{\text{true}}(k)|^2}
\right)
]

Results are averaged over **30 Monte-Carlo runs**.

---

## Noise Scenarios

### Gaussian noise

Baseline identification.

### Impulsive noise

[
n(k) = \alpha(k)A(k)
]

* rare large outliers
* tests the robustness of adaptive memory

---

## Results

![Experiment 4.1 — RLS vs RLSVF](results/experiment_4_1_reproduction.png)

### Parameter tracking

* RLSVF adapts faster after parameter jumps.
* Fixed-ρ RLS has slower convergence.

### Forgetting factor behaviour

* ( \rho(k) \downarrow ) when parameters change
* ( \rho(k) \rightarrow 1 ) in steady state

### NEE

* Similar steady-state accuracy in Gaussian noise
* Higher variance for RLSVF under impulsive noise (expected due to higher adaptive gain)

---

## How to Run

```bash
pip install -r requirements.txt
python src/rlsvf_experiment.py
```

---

## Key Concepts Demonstrated

* Recursive least squares as an **information-matrix update**
* Adaptive memory ( N = 1/(1-\rho) )
* Prediction error as a **change detector**
* Bias–variance–tracking trade-off in adaptive identification

---

## Reference

B. Kovačević, Z. Banjac, I. Kostić Kovačević
**Robust adaptive filtering using recursive weighted least squares with combined scale and variable forgetting factors**
EURASIP Journal on Advances in Signal Processing, 2016.
