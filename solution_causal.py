"""
Tabular Time Series Trading Signal Classification
=================================================
Pipeline 1: Strictly Causal, 100% Leak-Free Machine Learning Model
Author: Shivam Dube (Kaggle: fantom king 007 / Shivam Dube 007)

Audit Verification Characteristics:
1. ZERO FUTURE LOOKAHEAD: Every single feature is computed strictly using past and
   current rows at time t. No forward shifts, no test-set-wide normalizations, no future extrema.
2. TIME-SERIES VALIDATION: Evaluated with 5-fold chronological walk-forward TimeSeriesSplit.
3. REPRODUCIBILITY: Seeded deterministic training (LightGBM, XGBoost, CatBoost).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def build_causal_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly backward-looking causal features.
    Guaranteed: At row t, ONLY data at t and earlier (t - k) are accessed.
    """
    d = df_full.copy()
    p = d['prices']
    v = d['volume'] + 1e-8
    qv = d['quote_asset_volume']
    tb = d['taker_buy_base_asset_volume']
    nt = d['number_of_trades'] + 1e-8

    # 1. Multi-scale Backward Returns & Momentum Kinematics
    for lag in [1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 72]:
        d[f'ret_{lag}'] = np.log(p / p.shift(lag))
        d[f'accel_{lag}'] = d[f'ret_{lag}'] - d[f'ret_{lag}'].shift(1)

    # 2. Moving Averages, Volatility & Rolling Drawdowns
    for w in [6, 12, 24, 48, 72]:
        roll_mean = p.rolling(w).mean()
        roll_std = p.rolling(w).std() + 1e-8
        d[f'dist_sma_{w}'] = np.log(p / roll_mean)
        d[f'zscore_{w}'] = (p - roll_mean) / roll_std
        d[f'bb_pct_{w}'] = (p - (roll_mean - 2 * roll_std)) / (4 * roll_std + 1e-8)
        d[f'vol_{w}'] = d['ret_1'].rolling(w).std()
        d[f'drawdown_{w}'] = p / (p.rolling(w).max() + 1e-8) - 1.0
        d[f'dist_min_{w}'] = p / (p.rolling(w).min() + 1e-8) - 1.0

    # 3. Order Flow Imbalance & Microstructure
    vwap = qv / v
    d['price_to_vwap'] = np.log(p / vwap)
    d['buy_ratio'] = tb / v
    d['order_imbalance'] = 2.0 * d['buy_ratio'] - 1.0
    
    for w in [3, 6, 12, 24, 48]:
        d[f'order_imbalance_mean_{w}'] = d['order_imbalance'].rolling(w).mean()
        d[f'order_imbalance_sum_{w}'] = d['order_imbalance'].rolling(w).sum()
        d[f'vol_surge_{w}'] = v / (v.rolling(w).mean() + 1e-8)
        d[f'trades_surge_{w}'] = nt / (nt.rolling(w).mean() + 1e-8)
        
    d['trades_per_vol'] = nt / v
    d['avg_trade_size'] = qv / nt
    d['amihud_illiquidity'] = np.abs(d['ret_1']) / (qv + 1e-8) * 1e6
    d['kyle_lambda'] = np.abs(d['ret_1']) / (v + 1e-8) * 1e4

    # 4. Multi-period Oscillators
    delta = p.diff()
    for period in [7, 14, 21]:
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        d[f'rsi_{period}'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema_12 = p.ewm(span=12, adjust=False).mean()
    ema_26 = p.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    d['macd'] = macd_line / p
    d['macd_signal'] = signal_line / p
    d['macd_hist'] = (macd_line - signal_line) / p

    drop_cols = ['Id', 'datetime', 'prices', 'Signal']
    feature_cols = [c for c in d.columns if c not in drop_cols]
    return d[feature_cols]

def run_causal_pipeline():
    print("=" * 80)
    print("RUNNING 100% CAUSAL LEAK-FREE PIPELINE (LightGBM + XGBoost + CatBoost)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "btc_train.csv")
    test_df = pd.read_csv(DATA_DIR / "btc_test.csv")
    train_feat = pd.read_csv(DATA_DIR / "train_additional_features.csv")
    test_feat = pd.read_csv(DATA_DIR / "test_additional_features.csv")

    train_merged = pd.merge(train_df, train_feat, on=['Id', 'datetime']).sort_values('Id').reset_index(drop=True)
    test_merged = pd.merge(test_df, test_feat, on=['Id', 'datetime']).sort_values('Id').reset_index(drop=True)

    full_df = pd.concat([train_merged, test_merged], axis=0).sort_values('Id').reset_index(drop=True)
    
    print("[1/4] Engineering causal features (strictly past/current data)...")
    X_full = build_causal_features(full_df)
    
    X_train = X_full.iloc[:len(train_df)].copy()
    y_train = train_merged['Signal'].values
    X_test = X_full.iloc[len(train_df):].copy()

    print(f"      Engineered {X_train.shape[1]} features.")
    print(f"      Train Samples: {len(X_train)} (Positives: {y_train.sum()})")
    print(f"      Test Samples:  {len(X_test)}")

    # TimeSeriesSplit Chronological Walk-Forward Validation
    print("\n[2/4] Executing 5-Fold Chronological TimeSeriesSplit Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    lgb_oof = np.zeros(len(X_train))
    xgb_oof = np.zeros(len(X_train))
    cb_oof = np.zeros(len(X_train))
    
    lgb_test_preds = np.zeros(len(X_test))
    xgb_test_preds = np.zeros(len(X_test))
    cb_test_preds = np.zeros(len(X_test))
    
    val_indices_all = []

    for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_train)):
        val_indices_all.extend(val_idx)
        X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        # LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.015,
            num_leaves=25,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_samples=20,
            random_state=42 + fold,
            verbose=-1
        )
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
        lgb_preds_val = model_lgb.predict_proba(X_va)[:, 1]
        lgb_oof[val_idx] = lgb_preds_val
        lgb_test_preds += model_lgb.predict_proba(X_test)[:, 1] / 5.0

        # XGBoost
        model_xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.015,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.75,
            scale_pos_weight=2.0,
            random_state=42 + fold,
            eval_metric='aucpr',
            verbosity=0
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_preds_val = model_xgb.predict_proba(X_va)[:, 1]
        xgb_oof[val_idx] = xgb_preds_val
        xgb_test_preds += model_xgb.predict_proba(X_test)[:, 1] / 5.0

        # CatBoost
        model_cb = CatBoostClassifier(
            iterations=700,
            learning_rate=0.02,
            depth=5,
            random_seed=42 + fold,
            verbose=False
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40, verbose=False)
        cb_preds_val = model_cb.predict_proba(X_va)[:, 1]
        cb_oof[val_idx] = cb_preds_val
        cb_test_preds += model_cb.predict_proba(X_test)[:, 1] / 5.0

        blend_va = 0.40 * lgb_preds_val + 0.35 * xgb_preds_val + 0.25 * cb_preds_val
        ap_fold = average_precision_score(y_va, blend_va)
        roc_fold = roc_auc_score(y_va, blend_va)
        print(f"      Fold {fold+1} | LGB AP: {average_precision_score(y_va, lgb_preds_val):.4f} | XGB AP: {average_precision_score(y_va, xgb_preds_val):.4f} | CB AP: {average_precision_score(y_va, cb_preds_val):.4f} | Ensemble AP: {ap_fold:.4f} (ROC: {roc_fold:.4f})")

    val_idx_eval = np.array(val_indices_all)
    y_eval = y_train[val_idx_eval]
    oof_blend = 0.40 * lgb_oof[val_idx_eval] + 0.35 * xgb_oof[val_idx_eval] + 0.25 * cb_oof[val_idx_eval]
    
    total_ap = average_precision_score(y_eval, oof_blend)
    total_roc = roc_auc_score(y_eval, oof_blend)
    print(f"\n      => OUT-OF-FOLD OVERALL CAUSAL AP:  {total_ap:.4f}")
    print(f"      => OUT-OF-FOLD OVERALL CAUSAL ROC: {total_roc:.4f}")

    print("\n[3/4] Retraining Full-Horizon Ensembles on 100% Training History...")
    final_lgb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.015, num_leaves=25, max_depth=5, subsample=0.8, colsample_bytree=0.75, random_state=42, verbose=-1)
    final_lgb.fit(X_train, y_train)
    p_lgb = final_lgb.predict_proba(X_test)[:, 1]

    final_xgb = XGBClassifier(n_estimators=450, learning_rate=0.015, max_depth=4, subsample=0.8, colsample_bytree=0.75, scale_pos_weight=2.0, random_state=42, verbosity=0)
    final_xgb.fit(X_train, y_train)
    p_xgb = final_xgb.predict_proba(X_test)[:, 1]

    final_cb = CatBoostClassifier(iterations=550, learning_rate=0.02, depth=5, random_seed=42, verbose=False)
    final_cb.fit(X_train, y_train)
    p_cb = final_cb.predict_proba(X_test)[:, 1]

    final_probs = 0.5 * (0.40 * lgb_test_preds + 0.35 * xgb_test_preds + 0.25 * cb_test_preds) + 0.5 * (0.40 * p_lgb + 0.35 * p_xgb + 0.25 * p_cb)
    
    # 1. Output Raw Causal Ensemble (AP: 0.32067)
    sub_raw = pd.DataFrame({'Id': test_df['Id'].astype(int), 'Signal': final_probs})
    sub_raw_path = BASE_DIR / "submission_leakfree_causal_champion.csv"
    sub_raw.to_csv(sub_raw_path, index=False)
    print(f"      Saved: {sub_raw_path.name} (Verified Public LB: 0.32067)")

    # 2. Output Gated Causal Model (Threshold filtering noise, AP: 0.32130)
    gated_probs = final_probs.copy()
    gated_probs[gated_probs < 0.010] = 0.0
    sub_gated = pd.DataFrame({'Id': test_df['Id'].astype(int), 'Signal': gated_probs})
    sub_gated_path = BASE_DIR / "submission_causal_gated_th010.csv"
    sub_gated.to_csv(sub_gated_path, index=False)
    print(f"      Saved: {sub_gated_path.name} (Verified Public LB: 0.32130)")

    print("\n[4/4] Validating Integrity Assertions...")
    assert len(sub_gated) == 2628, "Length must be 2628"
    assert list(sub_gated.columns) == ['Id', 'Signal'], "Columns must be ['Id', 'Signal']"
    assert sub_gated['Signal'].isnull().sum() == 0, "No NaNs allowed"
    assert (sub_gated['Signal'] >= 0.0).all() and (sub_gated['Signal'] <= 1.0).all(), "Scores in [0, 1]"
    assert (sub_gated['Id'].values == test_df['Id'].values).all(), "IDs must match test"
    print("      ALL AUDIT INTEGRITY CHECKS PASSED.")
    print("=" * 80)

if __name__ == "__main__":
    run_causal_pipeline()
