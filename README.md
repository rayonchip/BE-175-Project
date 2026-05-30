# BE-176-Project

Aim 1: "How accurately can clinical variables predict CKD?"
→ This is a classification / model validation problem.
The workflow here is: fit a classifier (logistic regression is the natural starting point), then rigorously evaluate it.

Use k-fold cross-validation (not a single train/test split) to get an unbiased estimate of generalization error. For a disease classification task with likely class imbalance, use stratified k-fold to preserve class proportions in each fold.
Report accuracy, but also consider sensitivity/specificity — a model that predicts "no CKD" for everyone can have high accuracy if CKD is rare. Your exam vocab: bias-variance tradeoff, generalization error, overfitting.
If you have many clinical features relative to patients, regularization (ridge/lasso) is warranted to prevent overfitting — this is the high p, low n setting.


Aim 2: "Which biomarkers most strongly influence predictions?"
→ This is a feature importance / interpretability problem — PLSR is the star here.
This is almost exactly the Carroll et al. example from your 23Wkey. The key tool is PLSR with VIP scores:

PLSR handles the likely multicollinearity among clinical biomarkers (e.g., creatinine and GFR are correlated) by decomposing X into latent components that maximally covary with Y (CKD status).
VIP (Variable Importance of Projection) scores summarize each biomarker's contribution across all components. Scores > 1 are considered significant.
Exam red flag: don't say PCA here — PCA finds components of maximum variance in X ignoring Y. PLSR finds components that explain covariation between X and Y, which is what you want when the goal is prediction-relevant biomarkers.

For stability of those VIP scores across datasets → bootstrapping: repeatedly resample the data with replacement, refit PLSR, recalculate VIP scores, and report what fraction of bootstrap samples give VIP > 1 for each biomarker.

Aim 3: "How sensitive are model outputs to changes in inputs?"
→ This is sensitivity analysis, connected to regularization and model stability.
Two angles:

Condition number / ill-conditioned loss surfaces: if clinical features are highly correlated (multicollinear), small changes in inputs cause large swings in model parameters. This is exactly when ridge regression stabilizes predictions by shrinking coefficients — accepting some bias to reduce variance.
Bootstrap confidence intervals on coefficients: fit the model many times on resampled data; wide CI = unstable/sensitive; narrow CI = stable. This directly quantifies input sensitivity empirically.