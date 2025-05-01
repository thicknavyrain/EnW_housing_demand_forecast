

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # ╔══════ 10-year history, 5-year forecast, drivers & SHAP ══════╗
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, json, re, random

    PATHS = {
        "forecast":   "forecast_output.csv",
        "price_hist": "processed_data/house_prices_formatted.csv",
        "dwell":      "processed_data/net_additional_dwellings_cleaned.csv",
        "job":        "processed_data/job_density.csv",
        "pop":        "processed_data/population.csv",
    }

    # ——— pick random LA & tidy forecast row (same parsing logic as before) ———
    fcast = pd.read_csv(PATHS["forecast"])
    la    = random.choice(fcast.LA.unique())
    row   = fcast.loc[fcast.LA == la].iloc[0]

    records, shap_h1 = [], None
    for col, val in row.items():
        if col == "LA" or col.endswith("_feature_importance"):
            continue
        if m := re.match(r"(\d{4})_(.+?)_ci_(lower|upper)$", col):
            yr, tgt, side = int(m[1]), m[2], m[3]
            rec = next(r for r in records if r["Year"] == yr and r["Target"] == tgt)
            rec[f"CI_{side}"] = val
        elif m := re.match(r"(\d{4})_(.+)$", col):
            yr, tgt = int(m[1]), m[2]
            fi_json = json.loads(row[f"{yr}_{tgt}_feature_importance"])
            if (tgt, yr) == ("housing_price", 2026):
                shap_h1 = fi_json
            records.append(dict(Year=yr, Target=tgt, Point=val,
                                CI_lower=np.nan, CI_upper=np.nan))
    f_df = pd.DataFrame(records)

    # ——— helper for historical series 2016-25 ———
    def hist_series(path, la_code, name):
        df = pd.read_csv(path)
        la_col = [c for c in df.columns if c.lower().startswith("la")][0]
        yrs = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        s = (df.set_index(la_col).loc[la_code, yrs].astype(float)
               .rename(index=int).reindex(range(2016, 2026)).ffill())
        s.name = name; return s

    price_hist = (pd.read_csv(PATHS["price_hist"]).set_index("LA Code")
                    .loc[la].filter(regex=r"^\d{4}$").astype(float)
                    .rename(index=int).reindex(range(2016, 2026)).ffill())
    job_hist   = hist_series(PATHS["job"],   la, "Job density")
    dwell_hist = hist_series(PATHS["dwell"], la, "Net additional dwellings")
    pop_hist   = hist_series(PATHS["pop"],   la, "Population")

    # ——— figure layout ———
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13, 8), constrained_layout=False)

    # large left column, slimmer right column; small gap (wspace)
    gs  = fig.add_gridspec(nrows=2, ncols=2,
                           width_ratios=[3, 1.2],
                           height_ratios=[3, 1.2],
                           wspace=0.08, hspace=0.32)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_shap  = fig.add_subplot(gs[1, 0])

    # sub-GridSpec for the three square panels on the right
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_r = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 1], hspace=0.45)
    ax_job, ax_dwell, ax_pop = [fig.add_subplot(gs_r[i, 0]) for i in range(3)]

    # ——— price panel ———
    ax_price.plot(price_hist.index, price_hist.values, lw=2, label="Historical")
    sub = f_df[f_df.Target == "housing_price"]
    ax_price.plot(sub.Year, sub.Point, lw=2, marker="o", label="Forecast")
    ax_price.fill_between(sub.Year, sub.CI_lower, sub.CI_upper,
                          alpha=.15, color="tab:blue", label="95 % CI")
    ax_price.set_title(f"{la}: house-price forecast")
    ax_price.set_ylabel("£")
    ax_price.legend()

    # ——— SHAP bar (horizon-1) ———
    if shap_h1:
        names  = [n.replace("_", " ").title() for n in shap_h1.keys()]
        values = list(shap_h1.values())
        ax_shap.barh(names, values, color="slategray")
        ax_shap.set_title("Impact on Price")
        ax_shap.set_xlabel("£")
        ax_shap.grid(alpha=.2, axis="x")

    # ——— right-hand contextual squares ———
    def style_small(ax, series, color):
        ax.plot(series.index, series.values, color=color, lw=1.8)
        ax.set_xlim(2016, 2025)
        ax.set_box_aspect(1)           # square
        ax.grid(alpha=.3)

    style_small(ax_job,   job_hist,   "tab:orange")
    ax_job.set_title("Job density")

    style_small(ax_dwell, dwell_hist, "tab:green")
    ax_dwell.set_title("Net additional dwellings")

    style_small(ax_pop,   pop_hist,   "tab:red")
    ax_pop.set_title("Population")

    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
