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
import pingouin as pg

from scipy.stats import ttest_ind, pointbiserialr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix


#config (to change per person)
CSV = Path("data/yone_soloq_100games.csv")#enter path here
RESULTS = CSV.parent / "results"
RESULTS.mkdir(exist_ok=True)

FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)

REMAKE_MINUTES = 7.0   #if a game is under 7 mins, i consider that within the remake window 
RANDOM_STATE = 42    #for reproducibility 


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
                 "t_stat": t_stat, "p_raw": p_raw, 
                 "hedges_g": float(pg.compute_effsize(w.dropna().values, l.dropna().values, eftype="hedges", paired=False))})
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


#now that we have multicollinearity to show overlapping variables, here are the cols we wish to keep from the vif file before
KEEP_COLS = [
    "Kills_pm", "Deaths_pm", "Assists_pm", "CS_pm",
    "DamageTaken_pm", "VisionScore_pm", "WardsKilled_pm",
    "TurretKills_pm", "InhibitorKills_pm",
]
dfm = df[["Win"] + KEEP_COLS].dropna().copy()
Z = df_z.loc[dfm.index, KEEP_COLS]
X= sm.add_constant(Z)
y = dfm["Win"].astype(int)



model = sm.Logit(y, X).fit(disp=0)
conf = model.conf_int(); conf.columns = ["ci_low","ci_high"]
out = pd.concat([model.params, model.pvalues, conf], axis=1)
out.columns = ["coef","p_value","ci_low","ci_high"]
out["odds_ratio"], out["or_ci_low"], out["or_ci_high"] = np.exp(out["coef"]), np.exp(out["ci_low"]), np.exp(out["ci_high"])
out.drop("const").sort_values("p_value").to_csv(RESULTS / "multivariate_logit_trimmed.csv")



#now time to find auc, roc, and youdens j to confirm prediction performance of model
#first split between training and testing (decided on a 80/20 split once implemented validation so training data isnt to small)
X, y = dfm[KEEP_COLS], dfm["Win"].astype(int)
all_Xtr, Xte, all_ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


#within the training, we want to split between actually training and model validation (decided on to be a 60/20 split within the 80% "training")
Xtr, Xval, ytr, yval = train_test_split(all_Xtr, all_ytr, test_size = (2/8), stratify = all_ytr, random_state = RANDOM_STATE)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
])
pipe.fit(Xtr, ytr)

#we want to get our Youden's J from validation set
#Since our youdens j is super conservative and has a low sensitivity rate, i'm explicitly adding a cuttoff for sensitivity to be 
#greater than 0.6 so our model is reasonable
yval_prob = pipe.predict_proba(Xval)[:,1]
auc_val = roc_auc_score(yval, yval_prob)
fpr_val, tpr_val, thres_val = roc_curve(yval, yval_prob)
fpr_val, tpr_val, thres_val = fpr_val[1:], tpr_val[1:], thres_val[1:] #want to avoid the inf threshold
youden = tpr_val - fpr_val
cutoff = (tpr_val >= 0.7)
if np.any(cutoff):  
    best_local = np.argmax(youden[cutoff])
    best_val_index = np.flatnonzero(cutoff)[best_local]
else:
    best_val_index = int(np.argmax(youden))
best_thres = float(thres_val[best_val_index])
print(f"Validation ROC-AUC: {auc_val:.3f}")
print(f"Best cutoff = {best_thres:.3f}, J = {youden[best_val_index]:.3f}")

#now we can see the values on the test given the validation results to test performance
#note that given the test roc may not provide the same threshold at all, we will use the confusion matrix to ensure that the thresholds align
yte_prob = pipe.predict_proba(Xte)[:,1]
auc_test = roc_auc_score(yte, yte_prob)
yte_pred = (yte_prob >= best_thres).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, yte_pred).ravel()
sens = tp/(tp+fn) if (tp+fn) else 0.0
spec = tn/(tn+fp) if (tn+fp) else 0.0
youden_test = sens + spec - 1
print(f"Test ROC-AUC: {auc_test:.3f}")
print(f"Test J (EXACT at {best_thres:.3f}) = {youden_test:.3f} | Sens={sens:.3f}, Spec={spec:.3f}")



fpr_test, tpr_test, _ = roc_curve(yte, yte_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr_test, tpr_test, label=f"AUC={auc_test:.2f}")
plt.plot([0,1],[0,1],"k--")
plt.scatter(1-spec, sens, c="red", label=f"Cutoff={best_thres:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# Probability distribution graph
plt.figure(figsize=(6,5))
plt.hist(yte_prob[yte==1], bins=12, density=True, alpha=0.5, label="Wins")
plt.hist(yte_prob[yte==0], bins=12, density=True, alpha=0.5, label="Losses")
plt.axvline(best_thres, linestyle="--", color="red", label=f"Cutoff={best_thres:.2f}")
plt.xlabel("Predicted P(Win)")
plt.ylabel("Density")
plt.title("Predicted Win Probabilities — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "predicted_probabilities.png", dpi=200, bbox_inches="tight")
plt.show()