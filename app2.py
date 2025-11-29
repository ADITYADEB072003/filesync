# app2.py (patched)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ---------- Config ----------
CSV_PATH = 'extracted_statement.csv'
REPORT_CSV = 'evaluation_report.csv'
CONF_MATRIX_PNG = 'confusion_matrix.png'
CLASS_REPORT_TXT = 'classification_report.txt'
BEST_MODEL_PKL = 'best_model.pkl'

# ---------- Utilities ----------

def load_and_clean(path):
    df = pd.read_csv(path)
    # keep only rows where first column (Info or Date) parses as date
    date_col = None
    for c in df.columns[:3]:
        # pick a likely date column
        if 'info' in c.lower() or 'date' in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df['__date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df[df['__date'].notna()].copy()

    # Standardize column names
    df = df.rename(columns=lambda s: s.strip())

    # clean numeric columns
    for col in ['Withdrawal', 'Deposit', 'Closing Balance']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace('₹', '', regex=False).str.replace('Rs', '', case=False, regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            df[col] = df[col].fillna(mean_val)
            print(f"Cleaned '{col}': mean={mean_val:.2f}")
        else:
            print(f"Warning: column '{col}' not found in CSV")
            df[col] = 0.0

    df = df.sort_values('__date').reset_index(drop=True)
    return df


def feature_engineer(df):
    # assume df has Withdrawal, Deposit, Closing Balance, __date
    df['Deposit'] = df['Deposit'].fillna(0.0)
    df['Withdrawal'] = df['Withdrawal'].fillna(0.0)
    df['net'] = df['Deposit'] - df['Withdrawal']
    # lag features
    df['closing_lag1'] = df['Closing Balance'].shift(1)
    df['closing_lag2'] = df['Closing Balance'].shift(2)
    # rolling sums
    df['net_rolling_3'] = df['net'].rolling(3, min_periods=1).sum()
    df['withdrawal_rolling_3'] = df['Withdrawal'].rolling(3, min_periods=1).sum()

    # drop first row(s) where lag is NaN
    df = df.dropna(subset=['closing_lag1']).reset_index(drop=True)

    # clip extreme outliers at 1st/99th percentiles for numeric features to reduce skew
    num_cols = ['Withdrawal', 'Deposit', 'net', 'closing_lag1', 'closing_lag2']
    for c in num_cols:
        if c in df.columns:
            low, high = np.nanpercentile(df[c], [1, 99])
            df[c] = np.clip(df[c], low, high)

    return df


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # tolerance bands
    tols = [0.05, 0.10, 0.15]
    tols_pct = {}
    for t in tols:
        denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
        tols_pct[f'Acc_±{int(t*100)}%'] = np.mean(np.abs(y_pred - y_true) <= t * denom) * 100
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, **tols_pct}


def bin_and_classify(y_true, y_pred, n_bins=5):
    # use quantile-based bins on y_true
    edges = np.quantile(y_true, np.linspace(0, 1, n_bins+1))
    edges = np.unique(edges)
    if len(edges) <= 1:
        return None, None, None
    y_true_bins = np.digitize(y_true, edges[1:-1], right=True)
    y_pred_bins = np.digitize(y_pred, edges[1:-1], right=True)
    cm = confusion_matrix(y_true_bins, y_pred_bins)
    acc = accuracy_score(y_true_bins, y_pred_bins) * 100.0
    creport = classification_report(y_true_bins, y_pred_bins, zero_division=0)
    return cm, acc, creport


# ---------- Main training flow ----------
if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = load_and_clean(CSV_PATH)
    df = feature_engineer(df)

    # features and target
    FEATURES = ['Withdrawal', 'Deposit', 'net', 'closing_lag1', 'closing_lag2', 'net_rolling_3', 'withdrawal_rolling_3']
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X = df[FEATURES].fillna(0.0).values
    y = df['Closing Balance'].values

    # Chronological train/test split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    results = []

    # 1) Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    res_lr = evaluate_regression(y_test, lr.predict(X_test))
    res_lr['name'] = 'LinearRegression'
    results.append(res_lr)
    print('\nLinearRegression done')

    # 2) SVR with GridSearchCV (scale features and target)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

    svr_param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1, 0.5]}
    base_svr = SVR(kernel='rbf')
    svr_gs = GridSearchCV(base_svr, svr_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    svr_gs.fit(X_train_s, y_train_s)
    print(f"SVR best params: {svr_gs.best_params_}")
    best_svr = svr_gs.best_estimator_
    # wrapper
    class SVRWrapper:
        def __init__(self, model, scaler_X, scaler_y):
            self.model = model
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
        def predict(self, Xraw):
            Xs = self.scaler_X.transform(Xraw)
            yp_s = self.model.predict(Xs)
            return self.scaler_y.inverse_transform(yp_s.reshape(-1,1)).ravel()

    svr_wrapper = SVRWrapper(best_svr, scaler_X, scaler_y)
    res_svr = evaluate_regression(y_test, svr_wrapper.predict(X_test))
    res_svr['name'] = 'SVR (RBF)'
    res_svr['best_params'] = svr_gs.best_params_
    results.append(res_svr)

    # 3) Random Forest with RandomizedSearchCV
    rf = RandomForestRegressor(random_state=42)
    rf_param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=12, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    print(f"RandomForest best params: {rf_search.best_params_}")
    best_rf = rf_search.best_estimator_
    res_rf = evaluate_regression(y_test, best_rf.predict(X_test))
    res_rf['name'] = 'RandomForest'
    res_rf['best_params'] = rf_search.best_params_
    results.append(res_rf)

    # Save evaluation report
    rows = []
    for r in results:
        rows.append({
            'Model': r.get('name'),
            'MSE': r.get('mse'),
            'RMSE': r.get('rmse'),
            'MAE': r.get('mae'),
            'R2': r.get('r2'),
            'Acc_±5%': r.get('Acc_±5%'),
            'Acc_±10%': r.get('Acc_±10%'),
            'Acc_±15%': r.get('Acc_±15%'),
            'Binned_Accuracy(%)': None
        })
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)
    print(f"Saved evaluation report to: {REPORT_CSV}")

    # Choose best model by RMSE
    best = min(results, key=lambda r: r['rmse'])
    print('\nBest model by RMSE:', best['name'])

    # compute binning/classification for best model
    if best['name'] == 'SVR (RBF)':
        y_pred_best = svr_wrapper.predict(X_test)
    elif best['name'] == 'RandomForest':
        y_pred_best = best_rf.predict(X_test)
    else:
        y_pred_best = lr.predict(X_test)

    cm, bacc, creport = bin_and_classify(y_test, y_pred_best, n_bins=5)
    if cm is not None:
        # save confusion matrix png
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix ({best["name"]})')
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, [f'bin_{i}' for i in range(cm.shape[0])], rotation=45)
        plt.yticks(tick_marks, [f'bin_{i}' for i in range(cm.shape[0])])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PNG, dpi=150)
        plt.show()
        print(f"Saved confusion matrix to: {CONF_MATRIX_PNG}")

        # save classification report
        with open(CLASS_REPORT_TXT, 'w', encoding='utf-8') as f:
            f.write(f"Model: {best['name']}\n\n")
            f.write(creport if creport is not None else '')
        print(f"Saved classification report to: {CLASS_REPORT_TXT}")

    # save best model object
    if best['name'] == 'SVR (RBF)':
        joblib.dump(best_svr, BEST_MODEL_PKL)
    elif best['name'] == 'RandomForest':
        joblib.dump(best_rf, BEST_MODEL_PKL)
    else:
        joblib.dump(lr, BEST_MODEL_PKL)
    print(f"Saved best model to: {BEST_MODEL_PKL}")

    print('\nAll done.')

# ---------- Interactive User Prediction ----------
print("\n--- Predict Closing Balance from User Withdrawal Input ---")
try:
    # load best model
    model = joblib.load(BEST_MODEL_PKL)
    print(f"Loaded model from {BEST_MODEL_PKL}")
except Exception as e:
    print(f"Could not load best model: {e}")
    model = None

# load full dataframe for feature engineering reference
try:
    df_full = load_and_clean(CSV_PATH)
    df_full = feature_engineer(df_full)
except:
    df_full = None

while True:
    user_in = input("Enter Withdrawal amount (or 'q' to quit): ").strip()
    if user_in.lower() in ['q', 'quit', 'exit']:
        print("Exiting prediction tool.")
        break

    try:
        w = float(user_in)
    except:
        print("Please enter a valid number.")
        continue

    # Build feature vector from latest row
    last = df_full.iloc[-1]
    sample = pd.DataFrame([{
        'Withdrawal': w,
        'Deposit': last['Deposit'],
        'net': last['Deposit'] - w,
        'closing_lag1': last['Closing Balance'],
        'closing_lag2': last['closing_lag1'],
        'net_rolling_3': last['net_rolling_3'],
        'withdrawal_rolling_3': last['withdrawal_rolling_3']
    }])

    X_user = sample[FEATURES].values
    pred = None

    # If SVR, scale first
    if isinstance(model, SVR):
        try:
            scX = joblib.load('scaler_X.pkl')
            scY = joblib.load('sc_y.pkl')
            Xs = scX.transform(X_user)
            ys = model.predict(Xs)
            pred = scY.inverse_transform(ys.reshape(-1,1)).ravel()[0]
        except:
            pred = model.predict(X_user)[0]
    else:
        pred = model.predict(X_user)[0]

    print(f"Predicted Closing Balance: {pred:.2f}\n")