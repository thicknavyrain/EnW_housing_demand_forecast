

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # ───────────────── imports ─────────────────
    import warnings, random, re, pathlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap, scipy
    from scipy import sparse
    import json
    import time
    from tqdm import tqdm
    from collections import defaultdict

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import RandomizedSearchCV, KFold
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    return (
        ColumnTransformer,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RandomizedSearchCV,
        TimeSeriesSplit,
        XGBRegressor,
        json,
        np,
        pathlib,
        pd,
        plt,
        random,
        re,
        scipy,
        shap,
        sparse,
        time,
        tqdm,
        warnings,
    )


@app.cell
def _(
    ColumnTransformer,
    LinearRegression,
    OneHotEncoder,
    Pipeline,
    RandomizedSearchCV,
    TimeSeriesSplit,
    XGBRegressor,
    np,
    pathlib,
    pd,
    plt,
    random,
    re,
    warnings,
):

    # ───────────────── config ─────────────────
    warnings.filterwarnings("ignore")
    random.seed(42); np.random.seed(42)

    DATA_DIR = pathlib.Path("processed_data")
    FILES = {
        "Price": DATA_DIR/"house_prices_formatted.csv",
        "Aff":   DATA_DIR/"affordability_ratios.csv",
        "Dwell": DATA_DIR/"net_additional_dwellings_cleaned.csv",
        "Job":   DATA_DIR/"job_density.csv",
        "Pop":   DATA_DIR/"population.csv"
    }
    LAG_P = 5
    TRAIN_END = 2011          # 2001-2011 train
    VAL_FEAT_YEAR = 2011            # features from this year only
    VAL_TARGET_YEARS = range(2012, 2017) 
    VAL_YEARS = range(2012, 2017)   # 2012-2016 val features
    TEST_END = 2021

    TARGETS   = {"Price": "Price", "AffRatio": "Aff"}   # price & affordability
    # ───────────────── helpers ─────────────────
    def to_long(path, val):
        df = pd.read_csv(path)
        la = next((c for c in df.columns if re.fullmatch("(?i)la\\s*code", c)), df.columns[0])
        yrs = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        out = df[[la] + yrs].melt(id_vars=[la], var_name="Year", value_name=val)
        out.columns = ["LA", "Year", val]
        out["Year"] = out["Year"].astype(int)
        out["LA"] = out["LA"].astype(str).str.strip()
        return out[out["LA"].str.startswith("E")]

    price = to_long(FILES["Price"], "Price")
    aff   = to_long(FILES["Aff"],   "Aff")
    dwell = to_long(FILES["Dwell"], "Dwell")
    job   = to_long(FILES["Job"],   "Job")
    pop   = to_long(FILES["Pop"],   "Pop")

    df = (price.merge(aff, on=["LA","Year"])
                .merge(dwell,on=["LA","Year"])
                .merge(job,  on=["LA","Year"])
                .merge(pop,  on=["LA","Year"])
                .sort_values(["LA","Year"])
                .reset_index(drop=True))

    # -------- numeric + BOTH-direction fill so no interior NaNs --------
    for col in ["Dwell","Job","Pop"]:
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                     .groupby(df["LA"]).apply(lambda s: s.ffill().bfill())
                     .reset_index(level=0, drop=True))

    # -------- keep LAs with full 2001-2021 after fill --------
    yrs_full = set(range(2001, TEST_END+1))
    df = df[df.groupby("LA")["Year"].transform(lambda s: yrs_full.issubset(set(s))).astype(bool)]

    print(f"✅  Modelling will use {df['LA'].nunique()} local authorities "
          f"with complete data for 2001-{TEST_END}.")

    # -------- add lags --------
    def addlags(col,pref):
        for l in range(1, LAG_P+1):
            df[f"{pref}_lag{l}"] = df.groupby("LA")[col].shift(l)
    addlags("Price","price"); addlags("Dwell","dwell")
    addlags("Job","job");     addlags("Pop","pop")

    lag_feats = [f"{p}_lag{l}"
                 for p in ("price","dwell","job","pop")
                 for l in range(1, LAG_P+1)] + ["Dwell","Job","Pop"]

    prep = ColumnTransformer([
        ("num","passthrough",lag_feats),
        ("cat",OneHotEncoder(handle_unknown="ignore"),["LA"])
    ])
    grid = {
        "model__n_estimators": np.arange(300,701,100),
        "model__learning_rate": np.linspace(0.05,0.25,5),
        "model__max_depth": np.arange(4,8),
        "model__subsample": np.linspace(0.7,1.0,4),
        "model__colsample_bytree": np.linspace(0.7,1.0,4)
    }

    # global containers
    all_errs, mae_rows, mape_rows = [], [], []

    def run_model(target_col, targ_name):
        for h in range(1, 6):                       # horizons 1-5
            # ----- Training rows 2001-(2011-h) -----
            tr = df[df["Year"] <= TRAIN_END - h].copy()
            tr["y"] = df.groupby("LA")[target_col].shift(-h)
            tr = tr.dropna(subset=lag_feats + ["y"])
            if tr.empty:
                print(f"{targ_name} h={h}: no training rows"); continue

            # ----- Validation rows: feature year 2011 only -----
            val = df[df["Year"] == VAL_FEAT_YEAR].copy()
            val["y"] = df.groupby("LA")[target_col].shift(-h)
            val = val[val["Year"] + h <= VAL_TARGET_YEARS.stop - 1]   # keep targets ≤2016
            val = val.dropna(subset=lag_feats + ["y"])
            if val.empty:
                print(f"{targ_name} h={h}: no validation rows"); continue

            Xtr, ytr = tr[lag_feats+["LA"]], tr["y"]
            Xv,  yv  = val[lag_feats+["LA"]], val["y"]

            # cv = KFold(n_splits=3 if len(Xtr) >= 60 else 2,
            #             shuffle=True, random_state=42)
            cv = TimeSeriesSplit(n_splits=3)

            rs = RandomizedSearchCV(
                Pipeline([("prep", prep),
                          ("model", XGBRegressor(objective="reg:squarederror",
                                                 random_state=42))]),
                grid, n_iter=10, cv=cv, n_jobs=-1,
                scoring="neg_mean_absolute_error", random_state=42
            ).fit(Xtr, ytr)

            xgb = rs.best_estimator_
            ols = Pipeline([("prep", prep), ("model", LinearRegression())]).fit(Xtr, ytr)

            # -------- feature importance (top-10, no one-hots) --------
            fn  = xgb.named_steps["prep"].get_feature_names_out()
            imp = xgb.named_steps["model"].feature_importances_
            imp_df = (pd.DataFrame({"f": fn, "g": imp})
                        .query("~f.str.startswith('cat__')", engine="python")
                        .sort_values("g", ascending=False)
                        .head(10))
            plt.figure(figsize=(6,4))
            plt.barh(imp_df["f"][::-1], imp_df["g"][::-1])
            plt.title(f"{targ_name}  –  horizon {h} (top 10)")
            plt.tight_layout(); plt.show()

            # -------- predictions & per-row errors --------
            pred_x, pred_o = xgb.predict(Xv), ols.predict(Xv)
            ae_x,  ae_o    = np.abs(yv - pred_x), np.abs(yv - pred_o)
            mape_x, mape_o = ae_x / yv, ae_o / yv

            mae_rows.append({"Target": targ_name, "H": h,
                             "XGB_MAE": ae_x.mean(), "OLS_MAE": ae_o.mean()})
            mape_rows.append({"Target": targ_name, "H": h,
                              "XGB_MAPE": mape_x.mean(), "OLS_MAPE": mape_o.mean()})

            for la, ax, ao, mx, mo in zip(val["LA"], ae_x, ae_o, mape_x, mape_o):
                all_errs.append({"Target": targ_name, "H": h, "LA": la,
                                 "XGB_AE": ax, "OLS_AE": ao,
                                 "XGB_APE": mx, "OLS_APE": mo})

    # ---------- run both targets ----------
    run_model("Price", "Price")
    run_model("Aff",   "AffRatio")

    # ---------- summary tables ----------
    mae_df  = pd.DataFrame(mae_rows)
    mape_df = pd.DataFrame(mape_rows)
    print("\nMAE summary\n",  mae_df.pivot(index="H", columns="Target").round(1))
    print("\nMAPE summary\n", mape_df.pivot(index="H", columns="Target").round(3))

    # ---------- distribution plots ----------
    err_df = pd.DataFrame(all_errs)
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    for h in range(1,6):
        axes[0,0].boxplot(err_df.query("H==@h & Target=='Price'")["XGB_AE"],
                          positions=[h], widths=0.6)
        axes[0,1].boxplot(err_df.query("H==@h & Target=='Price'")["OLS_AE"],
                          positions=[h], widths=0.6)
        axes[1,0].boxplot(err_df.query("H==@h & Target=='Price'")["XGB_APE"],
                          positions=[h], widths=0.6)
        axes[1,1].boxplot(err_df.query("H==@h & Target=='Price'")["OLS_APE"],
                          positions=[h], widths=0.6)
    axes[0,0].set_title("XGB absolute error (Price)");  axes[0,1].set_title("OLS absolute error (Price)")
    axes[1,0].set_title("XGB MAPE (Price)");            axes[1,1].set_title("OLS MAPE (Price)")
    for ax in axes.flatten():
        ax.set_xticks(range(1,6)); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()


    return LAG_P, df, grid, lag_feats, mape_df, prep


@app.cell
def _(
    Pipeline,
    RandomizedSearchCV,
    TimeSeriesSplit,
    XGBRegressor,
    df,
    grid,
    lag_feats,
    np,
    pd,
    plt,
    prep,
    scipy,
    shap,
):
        # ───────── make test forecasts for 2017-2021 ─────────
    TEST_FEAT_YEAR  = 2016                # features from 2016
    TEST_TARGET_YRS = range(2017, 2022)   # 2017-2021
    test_mae, test_mape, test_err_rows = [], [], []

    def final_test(target_col, targ_name):
        for h in range(1, 6):
            # ---------- training frame up to 2016-h ----------
            tr = df[df["Year"] <= TEST_FEAT_YEAR - h].copy()
            tr["y"] = df.groupby("LA")[target_col].shift(-h)
            tr = tr.dropna(subset=lag_feats + ["y"])
            if tr.empty:
                print(f"{targ_name} h={h}: no training rows"); continue

            # ---------- test frame: feature year 2016 only ----------
            te = df[df["Year"] == TEST_FEAT_YEAR].copy()
            te["y_true"] = df.groupby("LA")[target_col].shift(-h)   # 2017-21 truth
            te = te.dropna(subset=lag_feats + ["y_true"])
            if te.empty:
                print(f"{targ_name} h={h}: no test rows"); continue

            Xtr, ytr = tr[lag_feats+["LA"]], tr["y"]
            Xte, yte = te[lag_feats+["LA"]], te["y_true"]

            cv = TimeSeriesSplit(n_splits=3)
            rs = RandomizedSearchCV(
                Pipeline([("prep", prep),
                          ("model", XGBRegressor(objective="reg:squarederror",
                                                 random_state=42))]),
                grid, n_iter=10, cv=cv, n_jobs=-1,
                scoring="neg_mean_absolute_error", random_state=42
            ).fit(Xtr, ytr)

            best_xgb = rs.best_estimator_
            preds    = best_xgb.predict(Xte)

            # ---------- feature-importance (top-10, no LA one-hots) ----------
            fn  = best_xgb.named_steps["prep"].get_feature_names_out()
            imp = best_xgb.named_steps["model"].feature_importances_
            imp_df = (pd.DataFrame({"f": fn, "g": imp})
                        .query("~f.str.startswith('cat__')", engine="python")
                        .sort_values("g", ascending=False)
                        .head(10))

            plt.figure(figsize=(6,4))
            plt.barh(imp_df["f"][::-1], imp_df["g"][::-1])
            plt.title(f"{targ_name} – horizon {h} (top 10 importance)")
            plt.tight_layout(); plt.show()

    # 1. transform Xte with the fitted ColumnTransformer
            Xte_mat = best_xgb.named_steps["prep"].transform(Xte)
            feat_names_full = best_xgb.named_steps["prep"].get_feature_names_out()

            # --- convert sparse → dense DataFrame so SHAP can measure len() ---
            if scipy.sparse.issparse(Xte_mat):
                Xte_mat = pd.DataFrame(Xte_mat.toarray(), columns=feat_names_full)
            else:
                Xte_mat = pd.DataFrame(Xte_mat, columns=feat_names_full)

            # 2. build explainer on the raw booster
            explainer = shap.TreeExplainer(best_xgb.named_steps["model"])
            shap_values = explainer(Xte_mat, check_additivity=False)

            # 3. pick top non–one-hot feature by mean |SHAP|
            shap_df = pd.DataFrame({
                "feat": feat_names_full,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
            })

            mask = (
                ~shap_df["feat"].str.startswith("cat__") &      # drop LA one-hots
                ~shap_df["feat"].str.contains("price", case=False)   # drop price lags
            )
            top_feat = (shap_df[mask]
                        .sort_values("mean_abs_shap", ascending=False)
                        .iloc[0]["feat"])

    # --- 4. dependence plot, coloured by num__price_lag1 -------------
            colour_feat = "num__price_lag1"
            if colour_feat not in feat_names_full:
                print(f"⚠️  {colour_feat} not in feature list; using default colour.")
                colour_feat = None   # fallback to SHAP's automatic choice
        
            # 4. dependence plot (coloured by SHAP of the same feature)
            shap.dependence_plot(
                top_feat,
                shap_values.values,
                Xte_mat,                       # dense DataFrame of features
                feature_names=feat_names_full,
                interaction_index=colour_feat, # colour-gradient source
                alpha=0.6, dot_size=20,
                title=f"{targ_name} h={h} – dependency on {top_feat}"
            )

        
            ae   = np.abs(yte - preds)
            mape = ae / yte

            test_mae.append({"Target": targ_name, "H": h, "MAE": ae.mean()})
            test_mape.append({"Target": targ_name, "H": h, "MAPE": mape.mean()})

            for la, e, p in zip(te["LA"], ae, mape):
                test_err_rows.append({"Target": targ_name, "H": h, "LA": la,
                                      "AE": e, "APE": p})

    # run both targets
    final_test("Price",   "Price")
    final_test("Aff",     "AffRatio")

    # -------- summary tables --------
    mae_test_df  = pd.DataFrame(test_mae).pivot(index="H", columns="Target", values="MAE").round(1)
    mape_test_df = pd.DataFrame(test_mape).pivot(index="H", columns="Target", values="MAPE").round(3)
    print("\nTEST MAE 2017-2021\n",  mae_test_df)
    print("\nTEST MAPE 2017-2021\n", mape_test_df)

    # -------- distribution plots --------
    test_err_df = pd.DataFrame(test_err_rows)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))

    for h2 in range(1, 6):
        ax2[0].boxplot(test_err_df.query("H == @h2 and Target == 'Price'")["AE"],
                       positions=[h2], widths=0.6)
        ax2[1].boxplot(test_err_df.query("H == @h2 and Target == 'Price'")["APE"],
                       positions=[h2], widths=0.6)

    ax2[0].set_title("XGB absolute error – Price (test)")
    ax2[0].set_ylabel("£")

    ax2[1].set_title("XGB MAPE – Price (test)")
    ax2[1].set_ylabel("ratio")

    for a in ax2:
        a.set_xticks(range(1, 6))
        a.grid(alpha=.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    Pipeline,
    RandomizedSearchCV,
    TimeSeriesSplit,
    XGBRegressor,
    df,
    grid,
    lag_feats,
    prep,
    time,
):
    models   = {}          # key = (targetCol, horizon) → fitted pipeline
    train_sec = 0.0

    for tgt_col, last_y in [("Price", df.Year.max()),        # 2025
                            ("Aff",   df.Year.max() - 1)]:   # 2024
        for h3 in range(1, 6):
            t0 = time.perf_counter()

            train = df[df.Year <= last_y - h3].copy()
            train["y"] = df.groupby("LA")[tgt_col].shift(-h3)
            train = train.dropna(subset=lag_feats + ["y"])

            rs = RandomizedSearchCV(
                Pipeline([("prep", prep),
                          ("model", XGBRegressor(objective="reg:squarederror",
                                                 random_state=42))]),
                grid, n_iter=10, cv=TimeSeriesSplit(3),
                scoring="neg_mean_absolute_error", n_jobs=-1,
                random_state=42
            ).fit(train[lag_feats+["LA"]], train["y"])

            models[(tgt_col, h3)] = rs.best_estimator_
            train_sec += time.perf_counter() - t0
            print(f"Trained {tgt_col} horizon {h3}")

    print(f"\n🏁 Total model-training time: {train_sec:.1f} s")
    return (models,)


@app.cell
def _(
    LAG_P,
    df,
    json,
    lag_feats,
    mape_df,
    models,
    pd,
    shap,
    sparse,
    time,
    tqdm,
):
    # ╔══════════ Cell B : inference & CSV ══════════╗


    mape_lookup = {(r.Target, int(r.H)): r.XGB_MAPE
                   for _, r in mape_df.iterrows()}

    def dense(x): return x.toarray() if sparse.issparse(x) else x

    def feat_row(target_year, la):
        """
        Return feature vector for (la, target_year).
        If target_year row is missing, use the latest available row ≤ target_year
        and forward-fill contemporaneous numerics.
        """
        sub = df[(df.LA == la) & (df.Year <= target_year)]
        if sub.empty:
            raise ValueError(f"No data at all for {la} up to {target_year}")
        base = sub.iloc[-1].copy()          # latest ≤ target_year

        # forward-fill current-year numerics if we borrowed an older row
        base["Year"] = target_year          # tag with intended year
        for col in ["Dwell", "Job", "Pop"]:
            if pd.isna(base[col]):
                base[col] = sub[col].ffill().iloc[-1]

        # make sure every lag column is populated
        for p in ("price", "dwell", "job", "pop"):
            for l in range(1, LAG_P + 1):
                col = f"{p}_lag{l}"
                if pd.isna(base[col]):
                    src_year = target_year - l
                    src_val  = df.loc[(df.LA == la) & (df.Year == src_year),
                                       col.replace("_lag"+str(l), "")].values
                    if len(src_val):
                        base[col] = src_val[0]
        return base[lag_feats + ["LA"]]

    rows, inf_sec = [], 0.0
    las = df.LA.unique()
    print(f"Running inference for {len(las)} LAs …")

    for la in tqdm(las):
        row = {"LA": la}
        for tgt_column, tgt_name, last_year in [("Price", "housing_price", 2025),
                                             ("Aff",  "affordability_ratio", 2024)]:
            for h4 in range(1, 6):
                try:
                    model = models[(tgt_column, h4)]
                    Xf    = feat_row(last_year, la).to_frame().T

                    t02 = time.perf_counter()
                    y_hat = float(model.predict(Xf)[0])
                    Xdense = dense(model.named_steps["prep"].transform(Xf))
                    sv     = shap.TreeExplainer(model.named_steps["model"])(
                                Xdense, check_additivity=False
                             ).values[0]
                    inf_sec += time.perf_counter() - t02

                    mape = mape_lookup.get(
                        (tgt_column if tgt_column == "Price" else "AffRatio", h4), 0.15)
                    hw   = 1.96 * mape * abs(y_hat)

                    yr  = last_year + h4
                    pfx = f"{yr}_{tgt_name}"
                    row[pfx]              = round(y_hat, 1)
                    row[pfx+"_ci_lower"]  = round(y_hat - hw, 1)
                    row[pfx+"_ci_upper"]  = round(y_hat + hw, 1)

                    fn = model.named_steps["prep"].get_feature_names_out()
                    shap_df = pd.DataFrame({"feat": fn, "shap": sv})
                    row[pfx+"_feature_importance"] = json.dumps({
                        "net_additional_dwellings":
                            float(shap_df[shap_df.feat.str.contains("dwell")].shap.sum()),
                        "job_density":
                            float(shap_df[shap_df.feat.str.contains("job")].shap.sum()),
                        "population":
                            float(shap_df[shap_df.feat.str.contains("pop")].shap.sum())
                    })
                except Exception as e:
                    print(f"Skip {la} {tgt_name} h={h4}: {e}")

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("forecast_output.csv", index=False)
    print(f"\n✅ Saved forecast_output.csv ({len(out_df)} rows)")
    print(f"🕒 Total inference + SHAP time: {inf_sec:.1f} s")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
