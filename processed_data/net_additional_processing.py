

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import re
    import matplotlib.pyplot as plt
    return pd, re


@app.cell
def _(pd):
    net_add_data = pd.read_csv('data_sources/net_additions.csv')
    net_add_data
    return (net_add_data,)


@app.cell
def _(net_add_data):
    print(net_add_data.head(10))
    return


@app.cell
def _(net_add_data, re):
    # Step 1: Strip " UA" from 'Authority Data' if it exists
    net_add_data['Authority Data'] = net_add_data['Authority Data'].str.replace(r'\s+UA$', '', regex=True)

    # Step 2: Clean and rename the year columns
    # Build a mapping from old to new column names
    rename_map = {}

    for col in net_add_data.columns:
        # Match columns that look like '2001-02' or '2001-02 [note 1]' etc.
        match = re.match(r'^(\d{4})-\d{2}', col)
        if match:
            rename_map[col] = match.group(1)
        # Match newer ones like '2021-22 [note 15]'
        match2 = re.match(r'^(\d{4})-\d{2}.*$', col)
        if match2:
            rename_map[col] = match2.group(1)

    # Apply renaming
    net_add_data_renamed = net_add_data.rename(columns=rename_map)

    print(net_add_data_renamed.columns.tolist())
    return (net_add_data_renamed,)


@app.cell
def _(net_add_data_renamed):
    net_add_data_renamed
    return


@app.cell
def _(net_add_data_renamed):
    print(net_add_data_renamed.head(10))
    return


@app.cell
def _(net_add_data_renamed):
    net_add_data_renamed.to_csv('processed_data/net_additional_dwellings_cleaned.csv')
    return


if __name__ == "__main__":
    app.run()
