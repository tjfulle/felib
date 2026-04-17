# Volumetric Locking: A Linear Algebra View

---

## 1. Start from Rank and Null Space

We solve:
\[
\mathbf{K}\mathbf{u} = \mathbf{f}
\]

For a Q4 element (plane strain or stress):

- DOFs: 8  
- Rigid body modes: 3  

\[
\dim \mathcal{N}(\mathbf{K}) = 3, \quad \text{rank}(\mathbf{K}) = 5
\]

This is the **correct baseline**.

---

## 2. Strain–Displacement Mapping

Strain is:
\[
\boldsymbol{\varepsilon} = \mathbf{B}\mathbf{u}
\]

Split into:
- Deviatoric part (shape change)
- Volumetric part:
\[
\varepsilon^{vol} = \text{tr}(\boldsymbol{\varepsilon})
\]

At a Gauss point \( g \):
\[
\varepsilon^{vol}_g = \mathbf{B}_{vol}^{(g)} \mathbf{u}
\]

---

## 3. Assemble the Discrete Divergence Operator

Define:
\[
\mathbf{C} =
\begin{bmatrix}
\mathbf{B}_{vol}^{(1)} \\
\mathbf{B}_{vol}^{(2)} \\
\mathbf{B}_{vol}^{(3)} \\
\mathbf{B}_{vol}^{(4)}
\end{bmatrix}
\]

- For fully integrated Q4: \( \mathbf{C} \in \mathbb{R}^{4 \times 8} \)  
- Each row = divergence evaluated at one Gauss point  

\[
\mathbf{C}\mathbf{u} =
\begin{bmatrix}
\varepsilon^{vol}_1 \\
\varepsilon^{vol}_2 \\
\varepsilon^{vol}_3 \\
\varepsilon^{vol}_4
\end{bmatrix}
\]

---

## 4. Constitutive Behavior and Energy

For nearly incompressible materials:
\[
\boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon}^{dev} + \kappa \, \text{tr}(\boldsymbol{\varepsilon}) \mathbf{I}
\]

Energy:
\[
\Pi = \int \left( 2\mu \, \varepsilon^{dev}:\varepsilon^{dev} + \kappa (\text{tr}\,\varepsilon)^2 \right) dV
\]

Discretized:
\[
\Pi \sim \mathbf{u}^T \mathbf{K}_{dev} \mathbf{u} + \kappa \|\mathbf{C}\mathbf{u}\|^2
\]

So:
\[
\mathbf{K} = \mathbf{K}_{dev} + \kappa \mathbf{C}^T \mathbf{C}
\]

---

## 5. Incompressible Limit

As:
\[
\kappa \to \infty
\]

The system enforces:
\[
\mathbf{C}\mathbf{u} \approx 0
\]

So admissible displacements lie in:
\[
\mathcal{U}_{adm} = \mathcal{N}(\mathbf{C})
\]

---

## 6. Key Question: What is \( \mathcal{N}(\mathbf{C}) \)?

This is the **discrete divergence-free space**.

### Continuous reality:
There are many smooth fields with:
\[
\nabla \cdot \mathbf{u} = 0
\]

### Discrete Q4 reality:
\[
\mathbf{C}\mathbf{u} = 0
\]
enforces this at **multiple points**

---

## 7. Constraint Counting (Critical Step)

For fully integrated Q4:

- \( \mathbf{C} \in \mathbb{R}^{4 \times 8} \)  
- rank(\( \mathbf{C} \)) ≈ 3  

So:
\[
\dim \mathcal{N}(\mathbf{C}) = 8 - 3 = 5
\]

Subtract rigid body modes (3):
→ **~2 deformational DOFs remain**

---

## 8. Compare to Physical Reality

Before incompressibility:
- deformational DOFs ≈ 5

After discrete incompressibility:
- deformational DOFs ≈ 2

👉 The space of admissible displacements has been **artificially shrunk**

---

## 9. Where Locking Comes From

The true solution requires:
- near divergence-free deformation
- but not exactly zero at all Gauss points

However:

\[
\mathbf{C}\mathbf{u} = 0
\]

is **too strict** for the Q4 interpolation space.

So:
- Most physically reasonable \( \mathbf{u} \) violate constraints  
- Violations are penalized by:
\[
\kappa \|\mathbf{C}\mathbf{u}\|^2
\]

As \( \kappa \to \infty \):
- even small violations → huge energy  
- solver suppresses deformation  

👉 **Artificial stiffness = locking**

---

## 10. What Happens to Rank

Important clarification:

- Rank of \( \mathbf{K} \) does **not drop to 1**
- It remains ≈ 5 (excluding rigid modes)

What changes:
- Eigenvalues split:
  - Deviatoric modes → moderate  
  - Volumetric modes → very large  

👉 Locking is a **spectral problem**, not a rank deficiency

---

## 11. Reduced Integration (Now Reinterpreted)

With 1 Gauss point:

\[
\mathbf{C}_{RI} =
\begin{bmatrix}
\mathbf{B}_{vol}^{(1)}
\end{bmatrix}
\in \mathbb{R}^{1 \times 8}
\]

- rank(\( \mathbf{C}_{RI} \)) = 1  

So:
\[
\dim \mathcal{N}(\mathbf{C}_{RI}) = 8 - 1 = 7
\]

Subtract rigid modes:
→ **~4 deformational DOFs**

---

## 12. Why This Fixes Locking

Compare:

| Case | Independent constraints | Deformational DOFs left |
|------|------------------------|-------------------------|
| Full integration | ~3 | ~2 |
| Reduced integration | 1 | ~4 |

👉 Reduced integration:
- Removes excess constraints  
- Makes \( \mathcal{N}(\mathbf{C}) \) large enough to contain good approximations  

So:
- \( \mathbf{C}\mathbf{u} \approx 0 \) becomes achievable  
- Energy does not blow up  
- No artificial stiffness  

---

## 13. Final Insight

> Volumetric locking occurs because the discrete divergence operator \( \mathbf{C} \) imposes **more independent constraints than the displacement space can satisfy**.

> Reduced integration works by **reducing the rank of \( \mathbf{C} \)** so that the constraint becomes **compatible with the interpolation space**.

---

## 14. One-Line Takeaway

> Locking is not “losing a DOF”—it is **overconstraining the discrete divergence**, and reduced integration fixes it by restoring compatibility between constraints and approximation space.