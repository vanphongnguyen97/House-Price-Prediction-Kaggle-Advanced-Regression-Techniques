# 🏠 House Price Prediction: Kaggle Advanced Regression Techniques

Predicting the sale price of residential homes in Ames, Iowa from 79 property features. The project covers outlier removal, feature engineering, a scikit-learn preprocessing pipeline, and comparison of linear, tree-based and boosting regressors, with a stacked ensemble used for the final submission.

🏆 **Kaggle public leaderboard score: 0.12458** (RMSE of log sale price)
🔗 Competition: [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Regression: predict `SalePrice` for each house in the test set |
| **Data** | 1,460 training houses and 1,459 test houses, 79 explanatory features |
| **Metric** | RMSE between log(predicted) and log(actual) price |
| **Tools** | Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost |
| **Final model** | Stacking ensemble (6 base models + weighted voting meta-learner) |

---

## 📂 Repository Structure

```
.
├── house_price_prediction.ipynb   
├── submission.csv                 # Final Kaggle submission
├── requirements.txt
└── README.md
```

> The data isn't included because of Kaggle's competition rules. Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

---

## 🔍 Methodology

### 1. Exploratory Data Analysis
- Scatter plots of every numeric feature against `SalePrice`, and box plots for categorical features.
- Extreme values flagged visually and with **Z-scores**. Examples include LotArea > 55,000 sq ft, LotFrontage > 300 and very large basements or garages.

### 2. Data Cleaning
- **Removed 21 outlier houses (1.4%)** that would distort the fit.
- **Missing values:** where `NaN` means "feature not present" (garage, basement, fireplace), it was filled with `"No"`. `LotFrontage` was filled with 0.
- **Dropped 7 sparse or redundant columns:** `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `GarageYrBlt` (highly correlated with `YearBuilt`), `GarageCond` and `BsmtFinType2`.
- Dropped `GarageArea` to avoid collinearity with `GarageCars`.

### 3. Feature Engineering
Six new features were created. The raw columns they combine were then dropped.

| Feature | Formula |
|---|---|
| `houseage` | YrSold − YearBuilt |
| `houseremodelage` | YrSold − YearRemodAdd |
| `totalsf` | 1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2 |
| `totalarea` | GrLivArea + TotalBsmtSF |
| `totalbaths` | BsmtFullBath + FullBath + 0.5 × (BsmtHalfBath + HalfBath) |
| `totalporchsf` | OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF |

**Target transform:** `SalePrice` → `log1p(SalePrice)` to correct right skew and to match the competition metric.

### 4. Preprocessing Pipeline (scikit-learn `ColumnTransformer`)

| Column group | Steps |
|---|---|
| Numeric | `SimpleImputer(mean)` → `StandardScaler` |
| Ordinal (quality/condition ratings, 18 cols) | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |
| Nominal (neighbourhood, style, etc., 19 cols) | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |

### 5. Modelling
Models were trained on an 80/20 split. Hyperparameters were tuned with **GridSearchCV** (scoring on negative MSE).

| Model | RMSE (log price) |
|---|---|
| **Ridge Regression** | **0.109** ✅ best single model |
| Gradient Boosting | 0.113 |
| CatBoost | 0.115 |
| XGBoost | 0.119 |
| LightGBM | 0.127 |
| Linear Regression | 0.133 |
| Random Forest | 0.134 |
| Voting ensemble (GBR + XGB + Ridge, weights 2:3:1) | 0.118 |
| **Stacking ensemble** (6 base models → voting meta-learner) | **0.118** → submitted |

Ridge outperformed the weakest model (Random Forest) by **18.5%**. After many features are one-hot encoded, the problem is close to linear, and regularisation handles the remaining collinearity well.

---

## 💡 Key Takeaways

- **Removing outliers and engineering combined area and bathroom features** gave the largest improvements.
- **The log-transformed target** made errors on cheap and expensive houses count equally, as the competition metric does.
- **Regularised linear models are hard to beat** on this dataset. Ridge matched or outperformed all boosting models.
- **Ensembling** combined diverse models for a stable final submission that scored **0.12455** on the leaderboard.

---

---

## 🙏 Acknowledgements

The Ames Housing dataset was compiled by [Dean De Cock](https://jse.amstat.org/v19n3/decock.pdf) for data science education.

---

## 👤 Author

**Van Phong Nguyen**
📧 phongnguyenvan97dn@gmail.com
