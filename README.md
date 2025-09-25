# Low-Elo League of Legends: Yone Match Outcome Prediction

This project analyzes 100 of my own ranked solo-queue Yone games played in low-elo (Iron-Gold range). 
The analysis begins with per-minute normalization, then univariate screening consisting of Welch t-tests with Cohen’s d, point-biserial correlations, and univariate logistic regression all with FDR correction, then multivariate logistic regression with VIF trimming, and finally model validation using ROC/AUC with Youden’s J and a sensitivity flloor constraint.

To set up the environment, create and activate a virtual environment and install the requirements in the command prompt:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
The dataset `yone_soloq_100games.csv` is already included in the `data` folder. 

To run the analysis, execute: 
```bash
python src/analysis.py 
```
The final paper with findings is a PDF in the `paper` folder.

**Final Results**
- Validation AUC: 0.865
- Validation Youden’s J: 0.708
- Validation cutoff: 0.662 (with sensitivity ≥ 0.7 floor)
- Test AUC: 0.812
- Test sensitivity: 0.583
- Test specificity: 0.875
- Test Youden’s J: 0.458
