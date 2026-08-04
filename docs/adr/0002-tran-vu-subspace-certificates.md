# ADR 0002: Tran--Vu subspace-stability certificates

- Status: accepted
- Date: 2026-07-27
- Decision owners: Grand Challenge Labs

## Context

Classical Davis--Kahan bounds scale the complete perturbation norm by the smallest target gap. That can be pessimistic when a perturbation is large globally but couples only weakly to the nearby signal modes. CPS already measures spectral motion, transient gain, eigenvalue conditioning, pseudospectral fragility, and projection closure, but it did not distinguish global perturbation strength from local directional coupling.

The reduced optimizer-state Jacobian is generally non-normal. A direct eigenspace application would therefore certify the wrong object. Tran and Vu provide a moderate-gap result for leading singular spaces of a general real matrix through symmetric dilation.

## Decision

CPS will compute both the classical Davis--Kahan comparison and the Tran--Vu Theorem 2.3 comparison for the leading singular spaces of each nontrivial coupling-phase sample relative to the nominal phase.

The implementation will:

1. use singular spaces as the default certified object for reduced operators;
2. compute the directional coupling exactly from the measured reference singular vectors and perturbation;
3. fail closed when the target gap, signal, halving rank, or moderate-gap hypotheses are unavailable;
4. mark a Tran--Vu result `admitted` only when the theorem applies, the bound is below one, and it is sharper than the classical bound;
5. preserve the observed projector displacement separately from both bounds;
6. map genuinely complex phase matrices to their real block representation and record the doubled working rank;
7. exclude duplicate nominal endpoints from sweep admission counts;
8. retain projection closure, non-normality, pseudospectral, and finite-horizon metrics as independent contracts.

The implementation also exposes the symmetric/Hermitian leading-eigenspace form of Theorem 2.1 for controlled operator studies, but the Pythia coupling pipeline uses the singular-space form.

## Consequences

The certificate is informative precisely when a small local gap is paired with weak perturbation coupling inside the nearby signal block. It will often fail or remain numerically vacuous; those null outcomes are reported rather than hidden.

Complex realification duplicates singular values and can increase the halving-rank penalty. This is accepted as the cost of preserving a theorem-backed real-matrix path for complex phase sweeps.

The directional coupling is deterministic once the two measured matrices are fixed. Sample splitting is not part of the certificate. It becomes necessary only when a statistical analysis interprets weak coupling as evidence that operator noise is independent of the reference singular vectors.

## Rejected alternatives

### Apply Davis--Kahan directly to non-normal eigenvectors

Rejected because the theorem controls invariant subspaces of symmetric/Hermitian operators, not the fragile right eigenvectors of a general non-normal Jacobian.

### Replace pseudospectral diagnostics with the new certificate

Rejected because singular-subspace rotation does not control resolvent growth or finite-horizon transient amplification.

### Admit every finite Tran--Vu expression

Rejected because the formula is theorem-backed only under the stated moderate-gap hypotheses, and a bound above one carries no useful projector-distance information.

## Reference

Phuc Tran and Van Vu, *Davis--Kahan Theorem under a Moderate Gap Condition*, arXiv:2510.22393, Theorems 2.1 and 2.3.
