# %% [markdown]
# # Predicción clínica: pCR, OS y DFS
# Pipeline reproducible: carga -> limpieza -> entrenamiento -> evaluación -> significancia.
# Ejecutable como script (`python pipeline_clinico.py`) o como notebook (jupytext / celdas `# %%`).

# %%
import warnings, os, itertools
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss)
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
OUT = "resultados"; os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## 1. CONFIGURACIÓN (lo único que debes editar)
# - `path`: archivo CSV/Excel de cada dataset.
# - `tasks`: por cada tarea, nombres de columnas.
#   - pCR: tarea binaria directa (`col`).
#   - OS / DFS: son supervivencia; se binarizan a un horizonte fijo (`horizon`, en las
#     mismas unidades que `time`): 1 = evento ocurrido dentro del horizonte,
#     0 = seguimiento >= horizonte sin evento. Los censurados antes del horizonte se excluyen
#     (no se conoce su etiqueta). Esto permite usar Accuracy/AUC/Brier como pide la actividad.
# - `drop`: columnas a eliminar (IDs, y TODA columna que filtre el desenlace: otras salidas).

# %%
DEMO = False
UP = ""  
DATASETS = {
    "pCR_train": {"path": UP + "train-data-pCR.csv",    "drop": [], "tasks": {"pCR": {"type": "binary", "col": "pCR"}}},
    "OS_data":   {"path": UP + "survival-data-OS.csv",  "drop": [], "tasks": {"OS":  {"type": "binary", "col": "OS"}}},
    "DFS_data":  {"path": UP + "survival-data-DFS.csv", "drop": [], "tasks": {"DFS": {"type": "binary", "col": "DFS"}}},
}

TEST_SIZE = 0.20
CV_SPLITS, CV_REPEATS = 5, 5      
SELECTION = "cv"                 
THRESHOLD = 0.5

# %% [markdown]
# ## 2. Datos (carga, limpieza y construcción de etiquetas)

# %%
def make_demo(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ID": range(n),
        "edad": rng.normal(52, 10, n),
        "tamano_tumor": rng.gamma(3, 10, n),
        "ER": rng.choice(["pos", "neg"], n),
        "HER2": rng.choice(["pos", "neg"], n),
        "grado": rng.choice([1, 2, 3], n),
        "Ki67": rng.uniform(0, 80, n),
    })
    df.loc[rng.random(n) < .05, "Ki67"] = np.nan
    z = 0.04*df.Ki67 - 0.03*df.tamano_tumor + (df.HER2 == "pos")*1.0 - 1
    df["pCR"] = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)
    risk = 0.03*df.tamano_tumor + 0.3*df.grado - (df.ER == "pos")*0.5
    for t in ("OS", "DFS"):
        df[f"{t}_time"] = rng.exponential(80/np.exp(risk - risk.mean()))
        df[f"{t}_event"] = (rng.random(n) < .6).astype(int)
    return df

def load(name, cfg):
    if DEMO:
        return make_demo(seed=hash(name) % 1000)
    p = cfg["path"]
    return pd.read_excel(p) if p.endswith((".xlsx", ".xls")) else pd.read_csv(p)

def build_xy(df, cfg, tname, tcfg):
    """Devuelve X (features limpias) e y (0/1) para una tarea."""
    df = df.copy()
    all_targets = {c for t in cfg["tasks"].values() for c in
                   (t.get("col"), t.get("time"), t.get("event")) if c}
    if tcfg["type"] == "binary":
        df = df[df[tcfg["col"]].notna()]
        y = df[tcfg["col"]]
        if y.dtype == object or str(y.dtype).startswith("str"):
            y = y.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
        y = y.astype(int)
    else:
        h, T, E = tcfg["horizon"], df[tcfg["time"]], df[tcfg["event"]]
        keep = T.notna() & E.notna() & ((E == 1) & (T <= h) | (T >= h))
        df, T, E = df[keep], T[keep], E[keep]
        y = ((E == 1) & (T <= h)).astype(int)
    X = df.drop(columns=[c for c in set(cfg["drop"]) | all_targets if c in df.columns])
    return X.reset_index(drop=True), y.reset_index(drop=True)


def preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

def get_models(X):
    boost = (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                           eval_metric="logloss", random_state=SEED, n_jobs=1)
             if HAS_XGB else HistGradientBoostingClassifier(random_state=SEED))
    raw = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "XGBoost" if HAS_XGB else "HistGB": boost,
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True,
                             max_iter=1000, random_state=SEED),
    }
    return {k: ThrClf(Pipeline([("prep", preprocessor(X)), ("clf", m)])) for k, m in raw.items()}



# %%
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

class ThrClf(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y):
        oof = cross_val_predict(clone(self.estimator), X, y, method="predict_proba",
                                cv=StratifiedKFold(3, shuffle=True, random_state=SEED))[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        f = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in grid]
        self.thr_ = float(grid[int(np.argmax(f))])
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.model_.classes_
        return self
    def predict_proba(self, X):
        return self.model_.predict_proba(X)

# %%
def all_metrics(y, p, thr=THRESHOLD):
    yh = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, yh),
            "Precision": precision_score(y, yh, zero_division=0),
            "Recall": recall_score(y, yh, zero_division=0),
            "F1": f1_score(y, yh, zero_division=0),
            "AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p)}

def cv_auc_f1(model, X, y):
    """AUC y F1 por fold en CV repetida estratificada (mismos folds para todos los modelos)."""
    rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    auc, f1 = [], []
    for tr, va in rskf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        auc.append(roc_auc_score(y.iloc[va], p))
        f1.append(f1_score(y.iloc[va], (p >= m.thr_).astype(int), zero_division=0))
    return np.array(auc), np.array(f1)


def nadeau_bengio(d, n_train, n_test):
    k = len(d); var = d.var(ddof=1)
    if var == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1/k + n_test/n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=k-1)

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx]); adj[idx] = min(1, run)
    return adj

def significance(best, scores, n_train, n_test):
    others = [m for m in scores if m != best]
    fr = stats.friedmanchisquare(*[scores[m] for m in scores]) if len(scores) >= 3 else None
    rows = []
    for o in others:
        d = scores[best] - scores[o]
        t, p_nb = nadeau_bengio(d, n_train, n_test)
        try: p_w = stats.wilcoxon(d).pvalue
        except ValueError: p_w = 1.0
        rows.append({"Mejor": best, "Comparado con": o, "Dif. media AUC": d.mean(),
                     "t (N-B)": t, "p (N-B)": p_nb, "p (Wilcoxon)": p_w})
    res = pd.DataFrame(rows)
    res["p_Holm (N-B)"] = holm(res["p (N-B)"].values)
    res["p_Holm (Wilcoxon)"] = holm(res["p (Wilcoxon)"].values)
    res["Significativo (Holm N-B<0.05)"] = (res["p_Holm (N-B)"] < 0.05) & (res["Dif. media AUC"] > 0)   # el mejor debe GANAR, no solo diferir
    return res, fr


def run():
    tables, sigs, summary = [], [], []
    for dname, cfg in DATASETS.items():
        df = load(dname, cfg)
        for tname, tcfg in cfg["tasks"].items():
            X, y = build_xy(df, cfg, tname, tcfg)
            print(f"\n=== {dname} | {tname} | n={len(y)} | prevalencia={y.mean():.2%} ===")
            if y.nunique() < 2 or y.value_counts().min() < CV_SPLITS * 2:
                print("  Clases insuficientes, se omite."); continue
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                                  random_state=SEED)
            models = get_models(X)
            test_rows, cv_auc, cv_f1 = [], {}, {}
            for mname, model in models.items():
                cv_auc[mname], cv_f1[mname] = cv_auc_f1(model, Xtr, ytr)   # solo sobre train
                model.fit(Xtr, ytr)
                m = all_metrics(yte, model.predict_proba(Xte)[:, 1], model.thr_)
                m["Umbral"] = model.thr_
                m.update({"CV_AUC_media": cv_auc[mname].mean(), "CV_AUC_sd": cv_auc[mname].std(), "CV_F1_media": cv_f1[mname].mean(),
                          "Dataset": dname, "Tarea": tname, "Modelo": mname})
                test_rows.append(m); print(f"  {mname:13s} AUC={m['AUC']:.3f} F1={m['F1']:.3f}")
            T = pd.DataFrame(test_rows)
            T["Score"] = (T["AUC"] + T["F1"]) / 2          # balance AUC/F1 (criterio clínico)
            key = "Score" if SELECTION == "test" else "CV_AUC_media"
            best = T.loc[T[key].idxmax(), "Modelo"]
            n_tr = len(ytr) * (CV_SPLITS - 1) / CV_SPLITS; n_te = len(ytr) / CV_SPLITS
            S, fr = significance(best, cv_auc, n_tr, n_te)
            S.insert(0, "Tarea", tname); S.insert(0, "Dataset", dname)
            sigs.append(S); tables.append(T)
            summary.append({"Dataset": dname, "Tarea": tname, "Mejor modelo": best, "Mejor por test": T.loc[T["Score"].idxmax(), "Modelo"],
                            "Friedman p": fr.pvalue if fr else np.nan,
                            "Supera significativamente a todos": bool(S["Significativo (Holm N-B<0.05)"].all()),
                            "Supera a (n de 4)": int(S["Significativo (Holm N-B<0.05)"].sum())})
            print(f"  >> Mejor: {best} | Friedman p={fr.pvalue:.4f}")
    pd.concat(tables).to_csv(f"{OUT}/metricas_test.csv", index=False)
    pd.concat(sigs).to_csv(f"{OUT}/significancia.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{OUT}/resumen_mejores.csv", index=False)
    return pd.concat(tables), pd.concat(sigs), pd.DataFrame(summary)

if __name__ == "__main__":
    T, S, R = run()
    print("\n", R.to_string(index=False))
