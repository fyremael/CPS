# GCL Research Programme Specification

## Programme

**Coupling-Phase Spectroscopy: Structured Pseudospectral Diagnostics for Optimizer Dynamics**

## Governing claim

A training optimizer should not be judged only by the nominal spectrum of its local update map. It should also be judged by how that spectrum and its finite-horizon amplification respond to structured changes in the orientation of internal couplings.

CPS turns that response into a measurable object and then tests whether it predicts, explains, and prevents training instability.

## 1. Object of study

Let the complete optimizer state be

\[
z_t=(\theta_t,s_t)\in\mathcal Z,
\]

where \(\theta_t\) contains model parameters and \(s_t\) contains optimizer state such as momentum, second moments, matrix statistics, or trust-region variables. With minibatch and randomness frozen at step \(t\), write one optimizer step as

\[
z_{t+1}=F_t(z_t).
\]

The local optimizer-state Jacobian is

\[
J_t=DF_t(z_t).
\]

Because \(J_t\) is too large to materialize, CPS operates on a reduced operator

\[
\widehat J_t=W_t^*J_tV_t,
\qquad W_t^*V_t=I_r.
\]

Orthogonal Galerkin projection uses \(W_t=V_t=Q_t\). Two-sided projection is preferred when non-normality is strong.

## 2. Phase-restricted structured pseudospectrum

For a nonzero reduced coupling \(a_{ij}\), define the magnitude-preserving family

\[
A^{(ij)}(\phi)
=A+(|a_{ij}|e^{i(\arg a_{ij}+\phi)}-a_{ij})e_ie_j^*.
\]

For angular budget \(\delta\), define

\[
\Sigma^{\mathrm{phase}}_{ij}(A;\delta)
=
\bigcup_{|\phi|\leq\delta}\sigma(A^{(ij)}(\phi)).
\]

The exact perturbation norm is

\[
\|A^{(ij)}(\phi)-A\|_2
=2|a_{ij}|\left|\sin\frac{\phi}{2}\right|.
\]

Hence

\[
\Sigma^{\mathrm{phase}}_{ij}(A;\delta)
\subseteq
\Lambda_{\varepsilon}(A),
\qquad
\varepsilon=2|a_{ij}|\sin(\delta/2),
\]

where \(\Lambda_\varepsilon(A)\) is the ordinary \(\varepsilon\)-pseudospectrum. CPS is therefore a restricted, interpretable slice through the full pseudospectrum.

## 3. Coordinate hierarchy

Scalar entry sweeps are basis-dependent. The programme therefore requires three levels.

### Level I: basis-aware scalar CPS

Use scalar couplings only in a basis with operational meaning: optimizer-state partitions, layer blocks, Krylov vectors, curvature modes, or recent-update modes.

### Level II: block-covariant CPS

For a coupling block \(B\), preserve its singular values while rotating singular channels:

\[
B(\boldsymbol\phi)
=U\,\mathrm{diag}(s_k e^{i\phi_k})V^*.
\]

This preserves the block operator norm and Frobenius norm. Internal unitary changes of coordinates alter the representation but not the singular values being held fixed.

### Level III: cycle-invariant moment spectroscopy

Measure Fourier coefficients of

\[
m_q(\phi)=\operatorname{tr}(A(\phi)^q).
\]

Trace moments are invariant under similarity and avoid eigenvalue branch-label ambiguity. Their phase harmonics reveal the participation and multiplicity of the selected coupling in closed walks.

## 4. Principal observables

For each coupling family, compute:

1. **Worst spectral radius**
   \[
   R_{ij}=\max_\phi \rho(A^{(ij)}(\phi)).
   \]

2. **Worst spectral abscissa** for continuous-time reductions.

3. **Finite-horizon gain**
   \[
   G_{ij}(K)=\max_\phi\max_{1\leq k\leq K}\|A^{(ij)}(\phi)^k\|_2.
   \]

4. **Kreiss surrogate**
   \[
   \widehat{\mathcal K}_{ij}
   =\max_{\phi,z\in\mathcal G}(|z|-1)\|(zI-A^{(ij)}(\phi))^{-1}\|_2.
   \]

5. **Minimum eigenvalue gap**
   \[
   \Delta_{ij}=\min_{\phi,k\neq\ell}|\lambda_k(\phi)-\lambda_\ell(\phi)|.
   \]

6. **Maximum eigenvalue condition number**
   \[
   \kappa_{ij}^{\max}
   =\max_{\phi,k}\frac{\|x_k\|\|y_k\|}{|y_k^*x_k|}.
   \]

7. **Loop geometry**, including displacement, length, and signed area.

8. **Trace-moment harmonics**, which are branch-invariant.

## 5. Planning contracts

CPS is a planning instrument only when it generates testable interventions.

### Damping contract

Select the least damping \(\gamma\) satisfying

\[
\max_{(i,j)\in\mathcal E}\max_\phi
\rho(\widehat J_t^{(ij)}(\phi;\gamma))
\leq 1-\mu
\]

and a finite-horizon reserve constraint

\[
\max_{(i,j)\in\mathcal E}G_{ij}(K;\gamma)\leq G_{\max}.
\]

### Momentum contract

Search candidate momentum values under matched nominal progress. Reject candidates whose augmented-state CPS risk exceeds the current setting.

### Preconditioner contract

Compare candidate preconditioners by both nominal conditioning and CPS risk. A preconditioner that improves Hessian condition number but increases non-normal transient gain fails the contract.

### Block-structure contract

Construct a directed coupling-risk graph over parameter or optimizer-state blocks. Merge blocks with high mutual CPS risk; split blocks with weak cross-coupling and high internal heterogeneity.

## 6. Hypotheses

**H1 — Predictive:** CPS metrics predict loss spikes or gradient explosions at positive lead time better than gradient norm, maximum Hessian eigenvalue, and nominal spectral radius alone.

**H2 — Discriminative:** Under matched nominal spectrum, CPS separates normal from non-normal update operators through transient observables.

**H3 — Causal:** CPS-selected damping and momentum reduce spike rate without materially degrading compute-to-target.

**H4 — Structural:** CPS-guided optimizer block partitions outperform size-based and layer-based partitions at equal state and compute budgets.

**H5 — Scalable:** Rank-64 to rank-256 projections preserve optimizer rankings and intervention decisions sufficiently well for practical use.

## 7. Experimental stages

### Stage A — Exact synthetic systems

- Quadratic objectives with exact update Jacobians.
- Normal and non-normal operators with matched eigenvalues.
- Directed acyclic path interference versus feedback-cycle spectral motion.
- Exact comparison against dense pseudospectra and transient gains.

### Stage B — Small neural systems

- MLP and small CNN.
- CIFAR-10 and synthetic regression.
- Optimizers: SGD, heavy-ball, AdamW, Shampoo-like block preconditioner.
- Full or near-full optimizer-state Jacobians where feasible.

### Stage C — Single-GPU transformer prototype

- 20M–150M decoder-only Transformer.
- TinyStories or a controlled FineWeb-Edu slice.
- Optimizers: AdamW, SGD with momentum, Muon-like matrix updates, Shampoo.
- Rank-32/64/128 projections.
- Sparse checkpoint cadence.

### Stage D — Scale transfer

- 350M–1B models when the single-GPU campaign justifies escalation.
- Test invariance of rankings under width, depth, batch size, and precision changes.

## 8. Baselines

Every predictive experiment must include:

- gradient norm and gradient-noise scale;
- update-to-weight ratio;
- largest Hessian or generalized Gauss–Newton eigenvalue;
- nominal spectral radius of the projected update operator;
- eigenvector condition number without phase sweeps;
- finite-horizon gain without phase sweeps;
- random structured perturbations matched in norm.

## 9. Acceptance gates

### Measurement gate

- Synthetic propositions recovered numerically.
- Eigenpair tracking passes branch-switch tests.
- Matrix-free projection agrees with dense projection on fixtures.
- Metric error below 5% at rank 64 on benchmark operators.

### Prediction gate

At least one CPS metric must improve area under the precision-recall curve for spike prediction by 10% relative over the strongest scalar baseline, with positive median lead time across at least five seeds.

### Intervention gate

A CPS-guided intervention must reduce instability incidence by at least 30% at matched token or example budget, while increasing time-to-target by no more than 5%.

### Scaling gate

Optimizer ranking and intervention direction must agree between prototype and scale-transfer models in at least 80% of checkpoint comparisons.

## 10. Failure conditions

The programme is falsified or materially weakened if:

- CPS adds no predictive information beyond nominal transient gain;
- scalar results vanish under reasonable basis changes and block CPS does not recover them;
- projected metrics are unstable with respect to rank or probe seed;
- intervention gains disappear under matched compute;
- phase families do not correspond to plausible uncertainty or design degrees of freedom.

## 11. Deliverables

- formal preprint;
- tested open-source prototype;
- optimizer-state functionalization layer;
- benchmark suite and immutable run records;
- CPS dashboard and coupling-risk maps;
- intervention controller with audit trail;
- negative-results ledger;
- scale-transfer report.

## 12. Pythia empirical subject contract

Pythia is the primary real-world subject because it supplies a common architecture across scale, longitudinal checkpoints, controlled data order, and native GPT-NeoX optimizer-state artifacts. The empirical ladder is governed as follows.

### 12.1 Subject ladder

| Gate | Subject | Role |
|---|---|---|
| P0 | Pythia-70M | selected-coordinate instrument validation |
| P1 | Pythia-70M, dense checkpoints | longitudinal CPS trajectory |
| P2 | PolyPythia-70M seeds | reproducibility and outlier prediction |
| P3 | PolyPythia-160M seed controls | initialization/data-order attribution |
| P4 | Pythia/PolyPythia-160M and 410M | scale-transfer test |
| P5 | Pythia-1B and 2.8B | sparse validation after smaller gates pass |

No result from P4 or P5 may be used to repair a policy selected after examining those subjects. Policies must be frozen at the preceding gate.

### 12.2 Complete optimizer-state coordinates

For AdamW the selected state is

\[
z=(\theta,m,u), \qquad u=\log(v+\tau),
\]

where \(\tau>0\) is a declared floor. The logarithmic coordinate preserves positivity of the second moment under local perturbation. A block-diagonal scaling \(S\) defines dimensionless coordinates and the analyzed operator

\[
\widetilde J=S J S^{-1}.
\]

Every evidence packet records the state layout, scaling rule, selected parameters, moment provenance, and projection basis. Native and reconstructed optimizer moments are separate evidence classes.

### 12.3 Selected-coordinate functionalization

The prototype differentiates only a governed subset of parameters while treating all other model parameters as frozen context. Let \(I\) select the probed coordinates. The local map is

\[
F_I:(\theta_I,m_I,u_I)\mapsto(\theta_I^+,m_I^+,u_I^+),
\]

with loss gradients evaluated by a functional model call. This is an exact Jacobian of the selected-coordinate map, not an assertion that the omitted coordinates are dynamically irrelevant. Projection closure residuals quantify leakage from the selected probe subspace.

### 12.4 Pythia hypotheses

**P-H1 — Prospective value.** At step \(t\), CPS features improve held-out prediction of a future disturbance over loss trend, gradient norm, update-to-weight ratio, nominal spectral radius, and nominal finite-horizon gain.

**P-H2 — Optimizer-state value.** The complete selected AdamW state outperforms a parameter-only or Hessian-only reduction.

**P-H3 — Seed stability.** Step-level CPS structure exceeds batch, projection, and numerical variance and is reproducible across PolyPythia seeds.

**P-H4 — Causal attribution.** Decoupled initialization/data-order runs exhibit separable CPS effects under a preregistered hierarchical analysis.

**P-H5 — Intervention value.** A CPS-selected real optimizer control improves a matched continuation under identical model state, data, and token budget.

### 12.5 Pythia acceptance gates

1. **Replay gate:** a loaded checkpoint and declared batch produce a deterministic baseline update.
2. **Derivative gate:** autodiff and centered finite-difference JVPs agree on fixtures and selected Pythia coordinates within a declared tolerance.
3. **Projection gate:** conclusions are stable under rank and random-probe ablation; closure residuals are reported.
4. **Prediction gate:** baseline+CPS improves held-out grouped AUC or average precision by a preregistered material margin.
5. **Intervention gate:** a frozen CPS policy improves matched continuation without exceeding the compute/time-to-target allowance.
6. **Transfer gate:** intervention direction and coupling-risk rank ordering remain useful at the next model scale.

### 12.6 Evidence boundary

The repository implements the instrument and all runner contracts. It does not represent unexecuted Colab campaigns as empirical evidence. A Pythia claim enters the manuscript only after the corresponding immutable run packet, execution log, environment manifest, and analysis decision record have been committed or archived by checksum.
