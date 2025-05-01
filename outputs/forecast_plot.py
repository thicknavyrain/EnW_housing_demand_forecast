

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # ╔════════════════  build & save PRICE + AFF plots  ═══════════════╗
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, re, json, random
    from pathlib import Path

    # ---------- paths ----------
    PATHS = {
        "forecast"   : "forecast_output.csv",
        "price_hist" : "processed_data/house_prices_formatted.csv",
        "aff_hist"   : "processed_data/affordability_ratios.csv",
        "dwell"      : "processed_data/net_additional_dwellings_cleaned.csv",
        "job"        : "processed_data/job_density.csv",
        "pop"        : "processed_data/population.csv",
        "la_index"   : "data_sources/LA_index.csv"                     # <-- new
    }

    # ---------- LA-name lookup ----------
    la_index = pd.read_csv(PATHS["la_index"])
    code_col = [c for c in la_index.columns if "code" in c.lower()][0]
    name_col = [c for c in la_index.columns if "LAD23NM" in c][0]
    LA_NAME  = dict(zip(la_index[code_col].astype(str).str.strip(),
                        la_index[name_col].astype(str).str.strip()))

    # ---------- helper: pull history 2016-end_year ----------
    def hist_series(path, la_code, nice_name, end_year):
        df = pd.read_csv(path)
        la_col = [c for c in df.columns if c.lower().startswith("la")][0]
        yrs = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        s = (df.set_index(la_col)
               .loc[la_code, yrs]
               .apply(pd.to_numeric, errors="coerce")
               .rename(index=int)
               .reindex(range(2016, end_year + 1))
               .ffill())
        s.name = nice_name
        return s

    # ---------- load wide forecast & pick random LA ----------
    fcast = pd.read_csv(PATHS["forecast"])
    la    = random.choice(fcast.LA.unique())
    la_name = LA_NAME.get(la, la)       # fall back to code if name missing
    row   = fcast.loc[fcast.LA == la].iloc[0]

    # ---------- tidy-up parser (point + CI + SHAP) ----------
    def parse_forecast_row(row, target_key):
        recs, shap_local = [], None
        for col, val in row.items():
            if col == "LA" or col.endswith("_feature_importance"):
                continue
            if m := re.match(r"(\d{4})_(.+?)_ci_(lower|upper)$", col):
                yr, tgt, side = int(m[1]), m[2], m[3]
                if tgt != target_key: continue
                r = next(r for r in recs if r["Year"] == yr)
                r[f"CI_{side}"] = val
            elif m := re.match(r"(\d{4})_(.+)$", col):
                yr, tgt = int(m[1]), m[2]
                if tgt != target_key: continue
                fi = json.loads(row[f"{yr}_{tgt}_feature_importance"])
                if ((target_key == "housing_price" and yr == 2026) or
                    (target_key == "affordability_ratio" and yr == 2025)):
                    shap_local = fi
                recs.append(dict(Year=yr, Point=val,
                                 CI_lower=np.nan, CI_upper=np.nan))
        return pd.DataFrame(recs), shap_local

    # ---------- build historical drivers ----------
    price_hist = hist_series(PATHS["price_hist"], la, "Price (£)", end_year=2025)
    aff_hist   = hist_series(PATHS["aff_hist"],  la, "Aff ratio",  end_year=2024)
    job_hist   = hist_series(PATHS["job"],       la, "Job density",  end_year=2022)
    dwell_hist = hist_series(PATHS["dwell"],     la, "Net additional dwellings", 2022)
    pop_hist   = hist_series(PATHS["pop"],       la, "Population",   end_year=2022)

    drivers = {"Job density": job_hist,
               "Net additional dwellings": dwell_hist,
               "Population": pop_hist}

    # ---------- function to make & save a figure ----------
    def build_save(target_key, hist_ser, y_label, file_stub):
        tidy, shap1 = parse_forecast_row(row, target_key)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(13, 8), constrained_layout=False)
        gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1.2],
                               height_ratios=[3, 1.2],
                               wspace=0.08, hspace=0.32)

        ax_main = fig.add_subplot(gs[0,0])
        ax_shap = fig.add_subplot(gs[1,0])
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs_r    = GridSpecFromSubplotSpec(3,1, subplot_spec=gs[:,1], hspace=0.45)
        ax_drv  = [fig.add_subplot(gs_r[i,0]) for i in range(3)]

        # --- main series + forecast ---
        ax_main.plot(hist_ser.index, hist_ser.values, lw=2, label="Historical")
        ax_main.plot(tidy.Year, tidy.Point, lw=2, marker="o", label="Forecast")
        ax_main.fill_between(tidy.Year, tidy.CI_lower, tidy.CI_upper,
                             alpha=.15, color="tab:blue", label="95 % CI")
        ax_main.set_title(f"{la_name}: {target_key.replace('_',' ').title()} Forecast")
        ax_main.set_ylabel(y_label); ax_main.legend()

        # --- SHAP bar ---
        if shap1:
            names  = [n.replace('_',' ').title() for n in shap1]
            vals   = list(shap1.values())
            ax_shap.barh(names, vals, color="slategray")
            ax_shap.set_title("Impact on Price" if "price" in target_key
                              else "Impact on Affordability Ratio")
            ax_shap.set_xlabel("£" if "price" in target_key else "Ratio units")
            ax_shap.grid(alpha=.2, axis='x')

        # --- driver panels ---
        for ax,(title,series),col in zip(ax_drv, drivers.items(),
                                         ['tab:orange','tab:green','tab:red']):
            ax.plot(series.index, series.values, color=col, lw=1.8)
            ax.set_title(title); ax.set_xlim(2016, 2025)
            ax.set_box_aspect(1); ax.grid(alpha=.3)

        out_dir = Path("plots"); out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"{la}_{file_stub}.png"
        fig.savefig(fname, bbox_inches='tight', dpi=300)
        plt.show(); plt.close(fig)
        print("Saved →", fname)

    # ---------- create & save both plots ----------
    build_save("housing_price",        price_hist, "£",     "price_forecast")
    build_save("affordability_ratio",  aff_hist,   "Ratio", "aff_forecast")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
