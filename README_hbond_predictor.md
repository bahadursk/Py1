# H-Bond Donor & Acceptor Count Prediction

Predicts the **number of hydrogen bond donors** and **hydrogen bond acceptors**
of a molecule from its SMILES string, using a Random Forest trained on 216
anticancer drug candidates from PubChem.

Part of a small series of SMILES → property prediction notebooks built on the
same `PubChem_compound_anticancer.csv` dataset (see also: XLogP prediction).

---

## Read this first: what this notebook actually is

Unlike XLogP (a proprietary empirical estimate that genuinely benefits from a
learned model), **H-bond donor/acceptor counts are exact, rule-based counts
directly computable from structure** — RDKit's `Lipinski.NumHDonors()` and
`Lipinski.NumHAcceptors()` give you the true answer instantly, with zero
uncertainty.

**In practice, use the direct RDKit calculation, not the ML model, whenever you
have a valid SMILES.** This notebook builds a predictive model anyway, for two
legitimate reasons:

1. **Teaching/consistency** — same pipeline structure as the other prediction
   notebooks in this series, useful as a worked example of what happens when
   you apply ML to something that doesn't really need it.
2. **Diagnostic value** — it reveals *how learnable* each property is from
   indirect structural features alone (see Results below), which is genuinely
   informative even though the model isn't the tool you'd deploy.

Every prediction in this notebook's output includes **both** the ML estimate
and the exact RDKit count, side by side, so this distinction is never hidden.

---

## What's in the notebook

| Step | What it does |
|---|---|
| 1. Setup | Installs RDKit, imports libraries, sets random seed |
| 2. Load data | Reads `PubChem_compound_anticancer.csv` from Colab's local storage |
| 3. Prepare targets | Keeps `H-Bond_Donor_Count` and `H-Bond_Acceptor_Count` (216/216 compounds have both — no missing labels) |
| 4. Compute features | 10 RDKit descriptors per molecule (MolWt, LogP, TPSA, ring counts, etc.) — deliberately **excludes** the exact donor/acceptor counting functions |
| 5. Split & scale | 80/20 train/test split; features standardized (fit on train only) |
| 6. Train model | Multi-output Random Forest (300 trees), evaluated with 5-fold cross-validation + held-out test set |
| 7. Predict unknowns | Enter any SMILES → get ML prediction *and* exact RDKit count |
| 8. Applicability domain | Flags predictions for molecules structurally outside the training range |
| 9. Save | Exports predictions CSV and triggers a Colab download |

---

## Requirements

- Google Colab (uses `google.colab.files` for download)
- `PubChem_compound_anticancer.csv` already present in Colab's local file storage
  (upload it via the folder icon in the left sidebar before running)
- No manual package installation needed — the first cell installs RDKit;
  pandas/scikit-learn/matplotlib are preinstalled on Colab

## How to run

1. Open the notebook in Colab.
2. Confirm `PubChem_compound_anticancer.csv` is visible in the file browser
   (left sidebar folder icon). Re-upload if the session has restarted.
3. Run all cells top to bottom (**Runtime → Run all**).
4. In the "Predict for unknown molecules" cell, replace the placeholder SMILES
   (aspirin, caffeine) with your own candidate molecule(s).
5. The final cell saves and downloads a CSV with your results.

## Output

`hbond_count_predictions.csv` with columns:

| Column | Meaning |
|---|---|
| `SMILES` | Input structure |
| `predicted_donor_count` / `predicted_acceptor_count` | ML model's estimate |
| `exact_donor_count` / `exact_acceptor_count` | True RDKit-calculated count — **use this one** |
| `in_applicability_domain` | `True`/`False` — whether the molecule's features fall within the training set's range |

---

## Results on this dataset

| Target | 5-fold CV R² | Held-out test R² | Held-out test RMSE |
|---|---|---|---|
| Acceptor count | 0.76 | 0.82 | ~1.1 |
| Donor count | 0.35 | 0.13 | ~1.2 |

**Acceptor count is reasonably learnable** from indirect features (TPSA and
heteroatom count carry most of the signal). **Donor count is not** — it hinges
on specific functional groups (–OH, –NH, –NH₂) that continuous structural
descriptors don't distinguish well once the exact counting function is
excluded. This asymmetry is expected and is itself the main insight this
notebook produces.

## Possible extensions

- Add fragment-based features (e.g. explicit counts of –OH, –NH, –NH₂ via
  RDKit SMARTS pattern matching) to see if donor-count R² improves.
- Compare against `GradientBoostingRegressor` or `XGBRegressor`.
- Retrain on a larger compound set as your database grows.
