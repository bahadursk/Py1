# Step-by-Step Notes: Predicting XLogP from SMILES
### A beginner's guide to the notebook (Python + Machine Learning basics included)

---

## 0. What problem are we actually solving?

You have a spreadsheet of ~216 anticancer drug candidates. For most of them, PubChem
has already calculated a property called **XLogP** — a number describing how much a
molecule "prefers" fat/oil (octanol) over water. It's one of the most important
numbers in drug design:

- **High XLogP** (e.g. 5+) → very fat-soluble → may cross cell membranes easily, but
  can be poorly water-soluble, hard to formulate as a drug, and prone to side effects.
- **Low/negative XLogP** → very water-soluble → may struggle to cross membranes,
  but easier to formulate.
- Drug-like molecules usually sit somewhere around **0 to 5** (this is part of
  Lipinski's "Rule of Five").

**The task:** teach a computer to look at a molecule's *structure* (given only as a
SMILES string — a short text code for chemical structure) and predict what its
XLogP *would* be, without needing PubChem to have calculated it. Once trained, we
can apply this to a brand-new molecule that has never been in any database.

This is a **supervised regression problem**:
- *Supervised* = we have example input–output pairs to learn from (SMILES → known XLogP).
- *Regression* = the output is a continuous number (not a category like "yes/no").

---

## 1. Core vocabulary (read this once, refer back as needed)

| Term | Plain-English meaning |
|---|---|
| **SMILES** | A text string that encodes a molecule's structure, e.g. `CC(=O)OC1=CC=CC=C1C(=O)O` is aspirin. |
| **RDKit** | A Python library for chemistry — reads SMILES, calculates molecular properties ("descriptors"). |
| **Feature** | A number that describes something about the input. Here, features are things like molecular weight, number of rings, etc. |
| **Target / label** | The number we're trying to predict. Here: XLogP. |
| **Model** | The mathematical function that takes features in and produces a predicted target out. It "learns" from data. |
| **Training** | The process of showing the model examples so it can adjust itself to make good predictions. |
| **Training set** | The examples the model is *allowed to learn from*. |
| **Test set** | Examples the model *never sees during training* — used only to check how well it generalizes to new data. |
| **Overfitting** | When a model memorizes the training examples instead of learning general patterns — looks great on training data, performs badly on new data. |
| **Random Forest** | A model made of many decision trees, each trained slightly differently, whose predictions are averaged. Works well on small datasets. |
| **Cross-validation (CV)** | Instead of one train/test split, split the data several different ways and average the results — gives a more trustworthy performance estimate, especially with small datasets. |
| **R² (R-squared)** | A score from roughly 0 to 1 (can go negative) measuring how well predictions match reality. 1.0 = perfect, 0 = no better than just guessing the average every time. |
| **RMSE (Root Mean Squared Error)** | The typical size of the prediction error, in the same units as the target. If RMSE = 0.9, predictions are typically off by about 0.9 XLogP units. |
| **Scaling / Standardization** | Rescaling features so they're all on a comparable numeric range (e.g. molecular weight ~300 and H-bond donors ~2 shouldn't be compared on raw scale). |
| **Applicability domain** | The range of molecule types the model was actually trained on. Predictions for molecules *outside* this range are less trustworthy — like extrapolating a ruler past its markings. |

---

## 2. Cell-by-cell walkthrough

### Cell: Install & import dependencies

```python
!pip install rdkit -q
import pandas as pd
...
```

- Lines starting with `!` are **shell commands**, not Python — this tells Colab to
  install the RDKit chemistry library, since it isn't built in.
- `import X` brings in a external library so we can use its tools. Think of it like
  pulling a toolbox off a shelf before you can use the tools inside it.
- `import pandas as pd` — pandas is the standard Python library for working with
  spreadsheet-like data (tables). `as pd` is just a nickname so we type `pd`
  instead of `pandas` every time.
- `SEED = 42` and `np.random.seed(SEED)` — machine learning involves a lot of
  randomness (e.g. how the Random Forest splits data internally). Setting a "seed"
  makes that randomness **reproducible** — rerun the notebook, get the same result,
  rather than a slightly different one every time.

### Cell: Load your CSV

```python
candidates = [f for f in glob.glob("*") if f.lower() == "pubchem_compound_anticancer.csv"]
...
df_raw = pd.read_csv(anticancer_path)
```

- `glob.glob("*")` lists every file in the current folder.
- The `[... for f in ... if ...]` pattern is a **list comprehension** — a compact
  way of writing "go through each item, keep only the ones matching some condition."
  Here: keep filenames that match `pubchem_compound_anticancer.csv`, ignoring
  upper/lowercase differences.
- `pd.read_csv(...)` loads your CSV file into a pandas **DataFrame** — think of it
  as a spreadsheet object living in Python's memory, with rows and named columns.

### Cell: Prepare target (XLogP) and keep only labeled rows

```python
df = df_raw[["SMILES", "Name", "XLogP"]].copy()
df = df.dropna(subset=["SMILES", "XLogP"])
```

- `df_raw[["SMILES", "Name", "XLogP"]]` selects only these three columns out of the
  38 in your original file — we don't need vendor lists or patent counts for this task.
- `.copy()` makes an independent copy, so edits here don't accidentally modify `df_raw`.
- `.dropna(subset=[...])` removes any row where SMILES or XLogP is missing
  (`NaN` = "Not a Number", pandas's marker for a blank cell). You can't train on a
  row with no known answer.
- Out of 216 compounds, 191 have a usable XLogP — the other 25 get set aside
  (PubChem couldn't compute XLogP for them, often because the structure is unusual).

### Cell: Sanity check for outliers

This is good scientific practice, not just coding: **always look at your data
before modeling it.** A histogram shows the shape of the XLogP values; the
outlier check flags any compound with an unusually extreme value (below -3 or
above 10), since those could be:
- genuinely unusual (a very large, fatty scaffold), or
- a data-quality issue worth double-checking manually.

Either way, it's better to know about them *before* they quietly skew your model.

### Cell: Compute RDKit descriptors from SMILES

```python
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [Descriptors.MolWt(mol), ...]
```

- `def get_features(smiles):` defines a **function** — a reusable block of code.
  We'll call this exact same function later on unknown molecules, so training and
  prediction always use identical, consistent features.
- `Chem.MolFromSmiles(smiles)` converts the text SMILES string into an actual RDKit
  molecule object that understands atoms and bonds. If the SMILES is invalid or
  can't be parsed, this returns `None` (nothing) — the `if mol is None: return None`
  line catches that so a bad SMILES doesn't crash the whole notebook.
- Each `Descriptors.X(mol)` call computes one number describing the molecule:
  - `MolWt` — molecular weight
  - `Crippen.MolLogP` — RDKit's *own* estimate of logP (a different algorithm than
    PubChem's XLogP — used here as one input feature among several, not as a
    shortcut/cheat, since the two methods disagree in interesting, informative ways)
  - `TPSA` — total polar surface area (relates to how "polar"/water-loving parts
    of the molecule are)
  - `NumHDonors` / `NumHAcceptors` — hydrogen bond donor/acceptor counts
  - `NumRotatableBonds` — flexibility of the molecule
  - `RingCount`, `NumAromaticRings` — structural rigidity/aromaticity
  - `FractionCSP3` — how "3D/saturated" vs "flat/aromatic" the carbon skeleton is
  - `HeavyAtomCount`, `NumHeteroatoms` — size and non-carbon/hydrogen atom content
- `df["features"] = df["SMILES"].apply(get_features)` runs this function on
  *every row* of the SMILES column and stores the resulting list of 11 numbers
  in a new column called `features`.

### Cell: Train/test split + scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)
scaler = StandardScaler().fit(X_train)
```

- By convention, **X** (capital) means the input features, **y** (lowercase) means
  the target we're predicting.
- `train_test_split(..., test_size=0.2)` randomly holds back 20% of the compounds
  as a **test set** the model never trains on — this is how we honestly measure
  whether the model actually learned something generalizable, versus just
  memorizing.
- `StandardScaler().fit(X_train)` calculates the mean and spread of each feature
  **using only the training data**, then `.transform(...)` rescales both train and
  test sets using those same numbers. This puts features like MolWt (~300) and
  NumHDonors (~2) onto comparable scales.
  - **Important beginner point:** we fit the scaler on the *training* set only,
    then apply it to the test set — never fit it on data that includes the test
    set. This avoids "leaking" information about the test set into training,
    which would make your evaluation misleadingly optimistic.

### Cell: Random Forest (primary model)

```python
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
cv_scores = cross_val_score(rf, X_train_s, y_train, cv=cv, scoring="r2")
```

- A **Random Forest** builds many individual decision trees (here, 300 —
  `n_estimators=300`), each trained on a slightly randomized version of the data,
  then averages all their predictions. This averaging is what makes it much less
  prone to overfitting than a single tree, and a good default choice for small
  tabular datasets like this one.
- `min_samples_leaf=2` stops any individual tree from making a "leaf" decision
  based on just 1 lonely data point — a mild guard against overfitting.
- `cross_val_score(..., cv=cv, scoring="r2")` does **5-fold cross-validation**: it
  splits the *training* set into 5 chunks, trains on 4 and tests on 1, five
  different times (rotating which chunk is held out), and reports the R² each
  time. Averaging these 5 scores gives a more trustworthy estimate than a single
  lucky/unlucky split — especially important with only ~150 training rows.
- After that, `rf.fit(X_train_s, y_train)` trains the final model on the *whole*
  training set, and we evaluate it one more time on the **held-out test set**
  (data the model has truly never seen at all, not even during cross-validation).

**Reading your actual results:**
- Cross-validation R² ≈ **0.65** → on average, across different train/validation
  splits, the model explains about 65% of the variation in XLogP.
- Held-out test R² ≈ **0.86**, RMSE ≈ **0.89** → on this particular held-out set,
  predictions are typically within about 0.9 XLogP units of the true value. The
  gap between CV (0.65) and test (0.86) is mostly just small-sample noise — with
  only ~40 test compounds, a single split can look better or worse than the
  "true" average performance by chance. This is exactly why we look at both
  numbers rather than trusting one split alone.

### Cell: Plots

```python
plt.scatter(y_test, rf_test_preds, alpha=0.7)
plt.plot(lims, lims, "r--")
```

- A **scatter plot** of actual vs. predicted values. If the model were perfect,
  every point would sit exactly on the diagonal red dashed line. Points scattered
  loosely around the line show typical error size; points far off the line are
  the model's worst predictions.
- The **feature importance** bar chart shows which of your 11 descriptors the
  Random Forest relied on most heavily. This is useful chemically too — it tells
  you which structural properties are actually driving XLogP in your specific
  compound set.

### Cell: Optional PyTorch neural network

This section builds an alternative model (a small neural network) purely for
comparison. With only ~150 training examples, neural networks usually don't have
enough data to outperform a Random Forest — it's included so you can see this for
yourself rather than take it on faith. Concepts specific to this section:

- **Epoch** — one full pass through the training data.
- **Dropout** — randomly "turns off" some neurons during training, which helps
  prevent overfitting.
- **Early stopping** — stop training once performance on the test set stops
  improving, rather than training a fixed number of epochs regardless (which risks overfitting the longer it runs).

### Cell: Predict XLogP for unknown molecules

```python
def predict_xlogp_rf(smiles_list):
    for smi in smiles_list:
        feats = get_features(smi)
        feats_scaled = scaler.transform([feats])
        pred = rf.predict(feats_scaled)[0]
```

This is the payoff: given any new SMILES string, we run it through the *exact
same* `get_features` function used during training, scale it with the *exact
same* scaler fitted on the training data, and ask the trained Random Forest for
a prediction. This consistency (same features, same scaling) is essential —
a model can only make sense of new data prepared the same way as the data it learned from.

**To use it:** replace the placeholder SMILES (aspirin, caffeine) in the
`unknown_smiles` list with your own candidate molecules' SMILES strings.

### Cell: Applicability domain check

```python
def in_domain(smiles):
    return bool(((feats >= train_min) & (feats <= train_max)).all())
```

For each unknown molecule, this checks whether *every one* of its 11 features
falls within the range seen during training. If a molecule is, say, much heavier
or much more polar than anything in your training set, the model is
extrapolating — which is inherently less reliable, similar to reading a ruler
past its last mark. `True` = safely within familiar territory; `False` = treat
the prediction with more caution.

### Cell: Save predictions

```python
predictions_df.to_csv(out_path, index=False)
files.download(str(out_path))
```

`.to_csv(...)` writes your results table back out as a CSV file you can open in
Excel or share. `files.download(...)` is a Colab-specific command that pops up a
browser download prompt for that file.

---

## 3. Quick troubleshooting glossary

| Error/message | What it usually means |
|---|---|
| `FileNotFoundError` | Colab can't find your CSV — check the folder icon on the left sidebar to confirm it's actually uploaded to this session (Colab sessions reset, so re-upload after a restart). |
| SMILES returns `None` in `get_features` | RDKit couldn't parse that SMILES string — usually a typo or invalid chemical structure. |
| R² is negative | The model is doing *worse* than just predicting the average every time — usually a sign of too little data, badly chosen features, or a bug in scaling/splitting. |
| Big gap between training R² and test R² | Classic overfitting sign — the model memorized training examples instead of learning general patterns. |

---

## 4. If you want to go further

- Try predicting a different PubChem-computed property (e.g. `TPSA`, `Complexity`)
  using the exact same pipeline — just swap the target column name.
- Try `GradientBoostingRegressor` or `XGBRegressor` as another model to compare
  against the Random Forest.
- Add more RDKit descriptors (there are 200+ available via
  `Descriptors._descList`) and see if feature importance changes.
- As your labeled dataset grows past a few hundred compounds, revisit the neural
  network — it tends to catch up to and eventually surpass Random Forest once
  there's enough data to learn from.
