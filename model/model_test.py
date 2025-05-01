

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import RandomizedSearchCV, KFold
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import warnings, random, time, math

    import matplotlib.pyplot as plt
    return (
        ColumnTransformer,
        KFold,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RandomizedSearchCV,
        XGBRegressor,
        np,
        pd,
        plt,
        random,
        warnings,
    )


@app.cell
def _(
    ARIMA,
    ColumnTransformer,
    KFold,
    LinearRegression,
    OneHotEncoder,
    Pipeline,
    RandomizedSearchCV,
    XGBRegressor,
    np,
    pd,
    plt,
    random,
    warnings,
):
    # ─────────────── 1. Load & merge ───────────────
    warnings.filterwarnings("ignore")
    random.seed(42)
    np.random.seed(42)

    hp = pd.read_csv("processed_data/house_prices_formatted.csv").set_index("LA Code")
    nd = pd.read_csv("processed_data/net_additional_dwellings_cleaned.csv")
    nd = nd[~nd["LA code"].isna()].set_index("LA code")

    hp_cols = [c for c in hp.columns if c.isdigit()]
    nd_cols = [c for c in nd.columns if c.isdigit()]

    hp_long = hp[hp_cols].reset_index().melt(id_vars=["LA Code"], var_name="Year", value_name="Price")
    nd_long = nd[nd_cols].reset_index().melt(id_vars=["LA code"], var_name="Year", value_name="Dwellings") \
                        .rename(columns={"LA code": "LA Code"})

    hp_long["Year"] = hp_long["Year"].astype(int)
    nd_long["Year"] = nd_long["Year"].astype(int)

    merged = (hp_long.merge(nd_long, on=["LA Code", "Year"], how="left")
                         .sort_values(["LA Code", "Year"])
                         .reset_index(drop=True))

    # numeric-ise dwellings + gap-fill
    merged["Dwellings"] = pd.to_numeric(merged["Dwellings"], errors="coerce")
    merged["Dwellings"] = merged.groupby("LA Code")["Dwellings"].transform(lambda s: s.ffill(limit=1))
    merged["Dwellings"] = merged["Dwellings"].fillna(
        merged.groupby("LA Code")["Dwellings"].transform("median")
    )

    # ─────────────── 2. Feature engineering (p = 8) ───────────────
    def add_lags(d, group, col, p, prefix):
        d = d.sort_values(["LA Code", "Year"]).copy()
        for lag in range(1, p + 1):
            d[f"{prefix}_lag{lag}"] = d.groupby(group)[col].shift(lag)
        return d

    p = 8
    df = merged[(merged["Year"] >= 2000) & (merged["Year"] <= 2019)].copy()
    df = add_lags(df, "LA Code", "Price", p, "price")
    df["dwell_lag0"] = df["Dwellings"]
    df = add_lags(df, "LA Code", "Dwellings", 1, "dwell")  # dwell_lag1

    price_lags  = [f"price_lag{i}" for i in range(1, p + 1)]
    dwell_lags  = ["dwell_lag0", "dwell_lag1"]
    feature_cols = price_lags + dwell_lags

    price_pivot = hp_long.pivot(index="LA Code", columns="Year", values="Price")

    # ─────────────── 3. Pipelines & grids ───────────────
    num_cat_prep = ColumnTransformer([
        ("num", "passthrough", feature_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["LA Code"])
    ])

    xgb_param_grid = {
        "model__n_estimators": np.arange(200, 801, 100),
        "model__learning_rate": np.linspace(0.03, 0.3, 10),
        "model__max_depth": np.arange(3, 9),
        "model__subsample": np.linspace(0.6, 1.0, 5),
        "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
    }

    base_xgb = XGBRegressor(
        objective="reg:squarederror", random_state=42,
        n_estimators=400, learning_rate=0.1, max_depth=5,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb_pipe = Pipeline([("prep", num_cat_prep), ("model", base_xgb)])
    ols_pipe = Pipeline([("prep", num_cat_prep), ("model", LinearRegression())])

    # ─────────────── 4. Loop over horizons ───────────────
    mae_rows, mape_rows, all_errs = [], [], []

    for h in range(1, 6):
        # ----- training rows up to (2014-h) -----
        train_rows = df[(df["Year"] <= 2014 - h)].dropna(subset=feature_cols).copy()
        train_rows["Target"] = train_rows.apply(
            lambda r: price_pivot.at[r["LA Code"], r["Year"] + h], axis=1)
        train_rows = train_rows.dropna(subset=["Target"])

        # ----- validation (features from 2014 only) -----
        val_rows = df[df["Year"] == 2014].dropna(subset=feature_cols).copy()
        val_rows["Target"] = val_rows.apply(
            lambda r: price_pivot.at[r["LA Code"], 2014 + h], axis=1)
        val_rows = val_rows.dropna(subset=["Target"])

        X_tr, y_tr = train_rows[feature_cols + ["LA Code"]], train_rows["Target"]
        X_val, y_val = val_rows[feature_cols + ["LA Code"]], val_rows["Target"]

        # ----- XGBoost + random search -----
        rs = RandomizedSearchCV(
            xgb_pipe, param_distributions=xgb_param_grid, n_iter=10,
            scoring="neg_mean_absolute_error", cv=KFold(2, shuffle=True, random_state=42),
            random_state=42, n_jobs=-1, verbose=0
        ).fit(X_tr, y_tr)
        best_xgb = rs.best_estimator_

        # ----- feature importance without one-hot columns -----
        fnames = best_xgb.named_steps["prep"].get_feature_names_out()
        importances = best_xgb.named_steps["model"].feature_importances_
        imp_df = (pd.DataFrame({"Feature": fnames, "Gain": importances})
                    .query("~Feature.str.startswith('cat__')", engine="python")
                    .sort_values("Gain", ascending=False)
                    .reset_index(drop=True))

        plt.figure(figsize=(7, 4))
        plt.barh(imp_df["Feature"].head(15)[::-1], imp_df["Gain"].head(15)[::-1])
        plt.title(f"XGB feature importance (horizon {h}) – top 15")
        plt.tight_layout(); plt.show()

        # ----- OLS baseline -----
        ols_pipe.fit(X_tr, y_tr)

        # ----- ARIMA benchmark -----
        arima_preds, arima_abs = [], []
        for la in val_rows["LA Code"]:
            # training series for this LA up to 2014-h
            series = price_pivot.loc[la, :2014 - h].astype(float).values
            best_mae, best_pred = np.inf, np.nan
            # tiny hyper-grid: (p,d,q) in {(1,1,0), (1,1,1), (2,1,1)}
            for order in [(1,1,0), (1,1,1), (2,1,1)]:
                try:
                    fc = ARIMA(series, order=order, trend="c").fit().forecast(steps=h)[-1]
                    mae = abs(fc - price_pivot.at[la, 2014 + h])
                    if mae < best_mae: best_mae, best_pred = mae, fc
                except: pass
            arima_preds.append(best_pred)
            arima_abs.append(best_mae)

        # ----- predictions & errors -----
        xgb_pred = best_xgb.predict(X_val)
        ols_pred = ols_pipe.predict(X_val)

        abs_xgb = np.abs(y_val - xgb_pred)
        abs_ols = np.abs(y_val - ols_pred)
        abs_arima = np.array(arima_abs)

        mape_xgb = abs_xgb / y_val
        mape_ols = abs_ols / y_val
        mape_arima = abs_arima / y_val

        # ----- store metrics -----
        mae_rows.append({"Horizon": h,
                         "XGB_MAE": abs_xgb.mean(),
                         "OLS_MAE": abs_ols.mean(),
                         "ARIMA_MAE": abs_arima.mean()})
        mape_rows.append({"Horizon": h,
                          "XGB_MAPE": mape_xgb.mean(),
                          "OLS_MAPE": mape_ols.mean(),
                          "ARIMA_MAPE": mape_arima.mean()})

        for la, ae_x, ae_o, ae_a, mp_x, mp_o, mp_a in zip(
            val_rows["LA Code"], abs_xgb, abs_ols, abs_arima, mape_xgb, mape_ols, mape_arima):
            all_errs.append({"LA": la, "H": h,
                             "XGB_AE": ae_x, "OLS_AE": ae_o, "ARIMA_AE": ae_a,
                             "XGB_APE": mp_x, "OLS_APE": mp_o, "ARIMA_APE": mp_a})

    # ─────────────── 5. Summaries ───────────────
    mae_tbl  = pd.DataFrame(mae_rows)
    mape_tbl = pd.DataFrame(mape_rows)
    print("\nMAE summary\n",  mae_tbl.to_string(index=False))
    print("\nMAPE summary\n", mape_tbl.to_string(index=False))

    # ─────────────── 6. Error-distribution plots ───────────────
    errs = pd.DataFrame(all_errs)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for idx, model in enumerate(["XGB", "OLS", "ARIMA"]):
        for h in range(1, 6):
            axs[0, idx].boxplot(errs[errs["H"] == h][f"{model}_AE"],
                                positions=[h], widths=0.6)
            axs[1, idx].boxplot(errs[errs["H"] == h][f"{model}_APE"],
                                positions=[h], widths=0.6)
        axs[0, idx].set_title(f"{model} absolute £ error")
        axs[1, idx].set_title(f"{model} MAPE")
        for ax in (axs[0, idx], axs[1, idx]):
            ax.set_xticks(range(1, 6))
            ax.grid(alpha=.3)

    plt.tight_layout(); plt.show()

    return


@app.cell
def _(mae_df):
    mae_df.head(10)
    return


@app.cell
def _(mape_df):
    mape_df.head(10)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
