# Extended mixed-domain comparison

The user requested Autoencoder, GNN, Transformer, GAN, LR, KNN, DT, RF, SVM, NB
and GB, then validation-based comparison of the best new candidate with the
previous MLP baseline. Exact KNN and an exact degree-2 polynomial kernel were
explicitly selected by the user. No ANN, RBF approximation, linear replacement
for kernel SVM, PCA reduction or training-row subsampling is authorized here.

All existing seed-42 training/validation/test memberships and training-fitted
native preprocessing remain unchanged. The 35,030,746 training rows retain their
natural domain proportions. All candidates use the four separate test sets.

Neural candidates use the previous domain feature-group encoder architecture:
supervised autoencoder with native reconstruction, a feature-relation message-
passing GNN (not a traffic/device topology), a two-layer transformer, and a
conditional GAN whose three-class discriminator supplies binary classification.
GAN feature vectors are auxiliary training samples, not adversarial network
attacks. Their generator is never applied to test data. Checkpoints use mean
validation macro-F1/AP, Adam .001, dropout .3, epoch limit 20, patience 3, batch
8192 and threshold .5. All full training rows appear in each completed epoch.

Classical pipelines use a full-information namespace: 153 native coordinates in
four disjoint domain blocks plus four domain indicators (157 coordinates).
Unavailable blocks are abstract zero coordinates, not measured network zeros.
No claim of common physical flow semantics is made. Different native adapters
mean these are comparisons of complete pipelines, not isolated classifier effects.

- LR: logistic regression on the exact native blocks; full-data stochastic Adam
  optimization with no hidden layers or dropout.
- SVM: complete algebraic degree-2 polynomial feature map for
  K(u,v)=(gamma*u.dot(v)+coef0)^2, gamma=1/157, coef0=1, C=1. The map is exact;
  finite primal hinge-loss optimization is not a certificate of a global optimum.
- KNN: k=5, uniform votes, exact Euclidean distance, KD-tree eps=0. Exploit
  block structure algebraically: within-domain exact searches plus other-domain
  minimum-norm candidates give the exact pooled neighbors. No domain is excluded.
  A full-dimensional orthogonal covariance rotation was investigated without
  whitening or truncation, but observed throughput was worse. Final evaluation
  resumes the original native-coordinate index and its saved prediction cursor.
  The slower rotated trial is retained separately; no accuracy outcome selected
  the index layout. Both search layouts preserve the same Euclidean metric.
- NB: Gaussian likelihood statistics over the complete block representation;
  all training rows, float64 accumulators, variance smoothing 1e-9.
- DT: depth 16, min_samples_leaf 50, max_features=sqrt, random_state 42.
- RF: 64 trees, depth 12, min_samples_leaf 50, max_features=sqrt, bootstrap with
  the full training population, two CPU workers. These resource caps differ from
  the paper's RF settings and will be reported explicitly.
- GB: XGBoost histogram gradient boosting, 300 rounds, depth 8, learning rate
  .1, row subsample 1, column subsample .8, max_bin 128. Training-only quantile cuts.

Candidate checkpoints/settings are frozen using validation before their own
test evaluations. The overall ranking reads validation scores only, even though
previous-stage test results have already been inspected. Rank new candidates by
mean validation macro-F1, then mean AP; exact new-candidate ties prefer smaller
saved models. Compare the new winner's validation pair with the prior MLP pair;
retain MLP on an exact pair tie. No new test result chooses the overall baseline.

New artifacts are isolated under `artifacts/expanded_neural`,
`artifacts/classical_native`, `artifacts/expanded_classical` and an extended report.
The existing MLP baseline/checkpoints and previous results are preserved. No
overall winner is claimed until all requested candidates complete and are audited.
