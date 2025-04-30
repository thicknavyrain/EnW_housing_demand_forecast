warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ---------- 1. Load + merge ---------- #
hp = pd.read_csv("processed_data/house_prices_formatted.csv").set_index("LA Code")
nd = pd.read_csv("processed_data/net_additional_dwellings_cleaned.csv")
nd = nd[~nd["LA code"].isna()].set_index("LA code")

hp_cols = [c for c in hp.columns if c.isdigit()]
nd_cols = [c for c in nd.columns if c.isdigit()]

hp_long = hp[hp_cols].reset_index().melt(id_vars=["LA Code"], var_name="Year", value_name="Price")
nd_long = nd[nd_cols].reset_index().melt(id_vars=["LA code"], var_name="Year", value_name="Dwellings").rename(columns={"LA code":"LA Code"})

hp_long["Year"] = hp_long["Year"].astype(int)
nd_long["Year"] = nd_long["Year"].astype(int)

merged = (hp_long.merge(nd_long, on=["LA Code","Year"], how="left")
                    .sort_values(["LA Code","Year"])
                    .reset_index(drop=True))

merged["Dwellings"] = pd.to_numeric(merged["Dwellings"], errors="coerce")

# Fill dwellings gaps
merged["Dwellings"] = merged.groupby("LA Code")["Dwellings"] \
                            .transform(lambda s: s.ffill(limit=1))
merged["Dwellings"] = merged["Dwellings"].fillna(
    merged.groupby("LA Code")["Dwellings"].transform("median")
)

# ---------- 2. Feature engineering (p=8) ---------- #
def add_lags(d, group, col, p, prefix):
    d = d.sort_values(["LA Code","Year"]).copy()
    for l in range(1, p+1):
        d[f"{prefix}_lag{l}"] = d.groupby(group)[col].shift(l)
    return d

p = 8
df = merged[(merged["Year"]>=2000) & (merged["Year"]<=2019)].copy()
df = add_lags(df, "LA Code", "Price", p, "price")
df["dwell_lag0"] = df["Dwellings"]
df = add_lags(df, "LA Code", "Dwellings", 1, "dwell")  # dwell_lag1

price_lags = [f"price_lag{i}" for i in range(1,p+1)]
dwell_lags = ["dwell_lag0", "dwell_lag1"]
feature_cols = price_lags + dwell_lags

# Pivot prices for target lookup
price_pivot = hp_long.pivot(index="LA Code", columns="Year", values="Price")

# Containers
mae_rows, mape_rows, all_errs = [], [], []

# Set up categorical+numeric preprocessor
preproc = ColumnTransformer([
    ("num", "passthrough", feature_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["LA Code"])
])

# Hyperparameter grid for XGB
param_distrib = {
    "model__n_estimators": np.arange(200, 801, 100),
    "model__learning_rate": np.linspace(0.03, 0.3, 10),
    "model__max_depth": np.arange(3, 9),
    "model__subsample": np.linspace(0.6, 1.0, 5),
    "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
}

# Prepare global OLS baseline pipeline (no tuning)
ols_pipe = Pipeline([
    ("prep", preproc),
    ("model", LinearRegression())
])

for h in range(1, 6):
    # ---------- Build training set (<= 2014-h) ----------
    train_rows = df[(df["Year"] <= 2014 - h)].dropna(subset=feature_cols).copy()
    train_rows["Target"] = train_rows.apply(
        lambda r: price_pivot.at[r["LA Code"], r["Year"]+h] if (r["Year"]+h) in price_pivot.columns else np.nan,
        axis=1
    )
    train_rows = train_rows.dropna(subset=["Target"])

    # ---------- Build validation (now-cast) set: Year==2014 ----------
    val_rows = df[df["Year"] == 2014].dropna(subset=feature_cols).copy()
    val_rows["Target"] = val_rows.apply(
        lambda r: price_pivot.at[r["LA Code"], r["Year"]+h] if (r["Year"]+h) in price_pivot.columns else np.nan,
        axis=1
    )
    val_rows = val_rows.dropna(subset=["Target"])

    X_train = train_rows[feature_cols + ["LA Code"]]
    y_train = train_rows["Target"]
    X_val   = val_rows[feature_cols + ["LA Code"]]
    y_val   = val_rows["Target"]

    # ---------- Define pipeline ----------
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=400,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0
    )
    pipe_xgb = Pipeline([
        ("prep", preproc),
        ("model", base_model)
    ])

    # ---------- Random search with 3-fold CV ----------
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    rand_search = RandomizedSearchCV(
        pipe_xgb,
        param_distributions=param_distrib,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_

    # 1. Get column names coming out of the ColumnTransformer
    feat_names = best_model.named_steps["prep"].get_feature_names_out()
    
    # 2. Pull importance scores from the XGBRegressor
    importances = best_model.named_steps["model"].feature_importances_
    
    # 3. Package into a DataFrame and sort
    imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
          .sort_values("Importance", ascending=False)
          .reset_index(drop=True)
    )
    
    # 4. Display / plot the top 20 (adjust N as needed)
    N = 20
    print(imp_df.head(N))
    
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["Feature"].head(N)[::-1], imp_df["Importance"].head(N)[::-1])
    plt.xlabel("Gain-based importance")
    plt.title("Top XGBoost feature importances")
    plt.tight_layout()
    plt.show()

    # ---------- Fit OLS baseline ----------
    ols_pipe.fit(X_train, y_train)

    # ---------- Predict on validation ----------
    xgb_preds = best_model.predict(X_val)
    ols_preds = ols_pipe.predict(X_val)

    abs_err_xgb = np.abs(y_val - xgb_preds)
    abs_err_ols = np.abs(y_val - ols_preds)
    mape_xgb = abs_err_xgb / y_val
    mape_ols = abs_err_ols / y_val

    mae_rows.append({
        "Horizon": h,
        "XGB_MAE": abs_err_xgb.mean(),
        "OLS_MAE": abs_err_ols.mean(),
        "Num_LAs": len(y_val)
    })
    mape_rows.append({
        "Horizon": h,
        "XGB_MAPE": mape_xgb.mean(),
        "OLS_MAPE": mape_ols.mean()
    })

    for la, ae_x, ae_o, ape_x, ape_o in zip(val_rows["LA Code"], abs_err_xgb, abs_err_ols, mape_xgb, mape_ols):
        all_errs.append({
            "Horizon": h,
            "LA Code": la,
            "XGB_AE": ae_x,
            "OLS_AE": ae_o,
            "XGB_APE": ape_x,
            "OLS_APE": ape_o
        })

# ---------- Display summary tables ----------
mae_df = pd.DataFrame(mae_rows)
mape_df = pd.DataFrame(mape_rows)

# ---------- Distribution plots ----------
err_df = pd.DataFrame(all_errs)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Absolute Error Distribution
for h in range(1,6):
    axes[0,0].boxplot(err_df[err_df["Horizon"]==h]["XGB_AE"], positions=[h], widths=0.6)
    axes[0,1].boxplot(err_df[err_df["Horizon"]==h]["OLS_AE"], positions=[h], widths=0.6)
axes[0,0].set_title("XGB Absolute Error (£)")
axes[0,1].set_title("OLS Absolute Error (£)")
axes[0,0].set_xticks(range(1,6)); axes[0,1].set_xticks(range(1,6))
axes[0,0].grid(alpha=0.3); axes[0,1].grid(alpha=0.3)
# MAPE Distribution
for h in range(1,6):
    axes[1,0].boxplot(err_df[err_df["Horizon"]==h]["XGB_APE"], positions=[h], widths=0.6)
    axes[1,1].boxplot(err_df[err_df["Horizon"]==h]["OLS_APE"], positions=[h], widths=0.6)
axes[1,0].set_title("XGB MAPE")
axes[1,1].set_title("OLS MAPE")
axes[1,0].set_xticks(range(1,6)); axes[1,1].set_xticks(range(1,6))
axes[1,0].grid(alpha=0.3); axes[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
