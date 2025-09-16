"""
Yone low-elo analysis (100 games)
-Normalize stats to be at a per minute rate
-Conduct univariate test: Welch's t-test with Cohen's d, point-biserial, and univariate logistic all with FDRs
-Conduct multivariate logistic regression: use all metrics first then use vif to trim metrics and then run it again
-Do a train/test split validation with ROC, AUC, and Youden's J to determine multivariate predictive power
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, pointbiserialr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

#config (to change per person)
CSV = Path("data/yone_soloq_100games.csv")#enter path here
RESULTS = CSV.parent / "results"
RESULTS.mkdir(exist_ok=True)

FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)

REMAKE_MINUTES = 7.0   #if a game is under 7 mins, i consider that within the remake window 
RANDOM_STATE = 42    #for reproducibility 


#helper function
#cohen's d doesn't exist in mainstreet libraries so i'll manually code it
def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna()
    y = y.dropna()
    if len(x) < 2 or len(y) <2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled_s = ((len(x)-1)*sx**2 + (len(y)-1)*sy**2) / (len(x)+len(y)-2)
    den = np.sqrt(pooled_s) if pooled_s > 0 else np.nan
    return (x.mean() - y.mean())/den if np.isfinite(den) else np.nan


#now import our csv and clean the data up
df = pd.read_csv(CSV)
df = df[df["Minutes"] > REMAKE_MINUTES].copy()
df = df.rename(columns = {
    "Minions Killed": "CS",
    "Damage Dealt": "DamageDealt",
    "Damage Taken": "DamageTaken",
    "Vision Score": "VisionScore",
    "Wards Placed": "WardsPlaced",
    "Wards Killed": "WardsKilled",
    "Turret Kills": "TurretKills",
    "Inhibitor Kills": "InhibitorKills"
})

#now change to per minute rates
orig_cols = ["Kills","Deaths","Assists","CS","Gold","DamageDealt","DamageTaken","VisionScore","WardsPlaced","WardsKilled","TurretKills","InhibitorKills"]

for c in orig_cols:
    df[f"{c}_pm"] = df[c] / df["Minutes"]

pm_cols = [c for c in df.columns if c.endswith("_pm")]

#now that data is cleaned we can move on to the univariate screening
#welch's t-test first
rows = []
for c in pm_cols:
    w = df.loc[df["Win"]==1, c]
    l = df.loc[df["Win"]==0, c]
    t_stat, p_raw = ttest_ind(w, l, equal_var=False, nan_policy="omit")
    rows.append({"metric": c, "mean_win": w.mean(), "mean_loss": l.mean(),
                 "t_stat": t_stat, "p_raw": p_raw, "cohens_d": cohens_d(w, l)})
ttest = pd.DataFrame(rows).sort_values("p_raw")
rej, p_fdr, _, _ = multipletests(ttest["p_raw"], alpha = 0.05, method = "fdr_bh")
ttest["p_fdr"], ttest["sig_fdr"] = p_fdr, rej
ttest.to_csv(RESULTS / "univariate_ttest.csv", index = False)


#now point biserial
rows = []
for c in pm_cols:
    r, p = pointbiserialr(df["Win"], df[c])
    rows.append({"metric": c, "r_pb": r, "p_raw": p})
pb = pd.DataFrame(rows)
rej, p_fdr, _, _ = multipletests(pb["p_raw"], alpha=0.05, method = "fdr_bh")
pb["p_fdr"], pb["sig_fdr"] = p_fdr, rej
pb.to_csv(RESULTS / "univariate_pb.csv", index = False)

#now univariate logit
#need to standardize our metrics to z values in order to have clear comparisons to one another
scaler = StandardScaler()
df_z = df.copy()
df_z[pm_cols] = scaler.fit_transform(df[pm_cols])


rows = []
for c in pm_cols:
    X = sm.add_constant(df_z[c])
    y = df_z["Win"].astype(int)
    model = sm.Logit(y,X).fit(disp = 0)
    coef, ci_low, ci_high  = model.params[c], *model.conf_int().loc[c]
    rows.append({"metric": c, "coef": coef, "odds_ratio": np.exp(coef),
                 "or_ci_low": np.exp(ci_low), "or_ci_high": np.exp(ci_high),
                 "p_raw": model.pvalues[c]})
uni_logit = pd.DataFrame(rows)
rej, p_fdr, _, _ = multipletests(uni_logit["p_raw"], alpha = 0.05, method = "fdr_bh")
uni_logit["p_fdr"], uni_logit["sig_fdr"] = p_fdr, rej
uni_logit.to_csv(RESULTS / "univariate_logit.csv", index = False)

#now multivariate logit
#we will reuse the standardized z from the univariate logic
Xcols = [c for c in df.columns if c.endswith("_pm")]
df_adj = df[["Win"] + Xcols].dropna().copy() 
Z = df_z.loc[df_adj.index, Xcols]
X = sm.add_constant(Z)
y = df_adj["Win"].astype(int)

#now multicollinearity using VIF
vif = pd.DataFrame({
    "feature": Z.columns,
    "VIF": [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]
}).sort_values("VIF", ascending=False)
vif.to_csv(RESULTS/ "vif.csv", index = False)


model = sm.Logit(y, X).fit(disp=0)
conf = model.conf_int(); conf.columns = ["ci_low","ci_high"]
out = pd.concat([model.params, model.pvalues, conf], axis=1)
out.columns = ["coef","p_value","ci_low","ci_high"]
out["odds_ratio"], out["or_ci_low"], out["or_ci_high"] = np.exp(out["coef"]), np.exp(out["ci_low"]), np.exp(out["ci_high"])
out.drop("const").sort_values("p_value").to_csv(RESULTS / "multivariate_logit.csv")


#now that we have multicollinearity to show overlapping variables, here are the cols we wish to keep 
KEEP_COLS = [
    "Kills_pm", "Deaths_pm", "Assists_pm", "CS_pm",
    "DamageTaken_pm", "VisionScore_pm", "WardsKilled_pm",
    "TurretKills_pm", "InhibitorKills_pm",
]
dfm = df[["Win"] + KEEP_COLS].dropna().copy()
Z = df_z.loc[dfm.index, KEEP_COLS]
X= sm.add_constant(Z)
y = dfm["Win"].astype(int)

vif = pd.DataFrame({
    "feature": Z.columns,
    "VIF": [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]
})
vif.to_csv(RESULTS / "vif_trimmed.csv", index=False)

model = sm.Logit(y, X).fit(disp=0)
conf = model.conf_int(); conf.columns = ["ci_low","ci_high"]
out = pd.concat([model.params, model.pvalues, conf], axis=1)
out.columns = ["coef","p_value","ci_low","ci_high"]
out["odds_ratio"], out["or_ci_low"], out["or_ci_high"] = np.exp(out["coef"]), np.exp(out["ci_low"]), np.exp(out["ci_high"])
out.drop("const").sort_values("p_value").to_csv(RESULTS / "multivariate_logit_trimmed.csv")



#now time to find auc, roc, and youdens curve to confirm prediction performance of model
X, y = dfm[KEEP_COLS], dfm["Win"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
])
pipe.fit(Xtr, ytr)

y_prob = pipe.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, y_prob)
fpr, tpr, thr = roc_curve(yte, y_prob)
print(f"Test ROC-AUC: {auc:.3f}")

youden = tpr - fpr
best_idx = int(np.argmax(youden))
best_thr = float(thr[best_idx])
print(f"Best cutoff = {best_thr:.3f}, J = {youden[best_idx]:.3f}")



#plot for ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0,1],[0,1],"k--")
plt.scatter(fpr[best_idx], tpr[best_idx], c="red", label=f"Cutoff={best_thr:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Trimmed Model")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.show()

#Probability distribution graph
plt.figure(figsize=(6,5))
plt.hist(y_prob[yte==1], bins=12, density=True, alpha=0.5, label="Wins")
plt.hist(y_prob[yte==0], bins=12, density=True, alpha=0.5, label="Losses")
plt.axvline(best_thr, linestyle="--", color="red", label=f"Cutoff={best_thr:.2f}")
plt.xlabel("Predicted P(Win)")
plt.ylabel("Density")
plt.title("Predicted Win Probabilities — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "predicted_probabilities.png", dpi=200, bbox_inches="tight")
plt.show()