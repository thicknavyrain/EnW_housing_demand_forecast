

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    data_spine = pd.read_csv('data_sources/LA_index.csv')
    return (data_spine,)


@app.cell
def _(data_spine):
    data_spine
    return


@app.cell
def _(pd):
    price_data = pd.read_csv('data_sources/house_price_index.csv')
    price_data
    return (price_data,)


@app.cell
def _(price_data):
    print(price_data.head())
    return


@app.cell
def _(pd, price_data):
    # Ensure Date column is datetime type
    price_data['Date'] = pd.to_datetime(price_data['Date'], dayfirst=True)

    # Extract year
    price_data['Year'] = price_data['Date'].dt.year

    # Group by LA Code and Year, then average the AveragePrice
    annual_avg_price = (
        price_data
        .groupby(['LA Code', 'Year'])['AveragePrice']
        .mean()
        .reset_index()
        .rename(columns={'AveragePrice': 'AnnualAveragePrice'})
    )

    # Optional: merge with region names if needed
    # This assumes RegionName doesn't change within an LA Code
    region_map = price_data[['LA Code', 'RegionName']].drop_duplicates()
    annual_avg_price = annual_avg_price.merge(region_map, on='LA Code', how='left')

    # Reorder columns
    annual_avg_price = annual_avg_price[['Year', 'RegionName', 'LA Code', 'AnnualAveragePrice']]

    # Rename RegionName to LA Name
    annual_avg_price = annual_avg_price.rename(columns={'RegionName': 'LA Name'})

    annual_avg_price
    return (annual_avg_price,)


@app.cell
def _(annual_avg_price):
    print(annual_avg_price.head())
    return


@app.cell
def _(annual_avg_price):
    # Pivot the table so years become columns
    pivoted = annual_avg_price.pivot(index=['LA Code', 'LA Name'], columns='Year', values='AnnualAveragePrice')

    # Reset index to turn LA Code and LA Name back into columns
    pivoted = pivoted.reset_index()

    # Optional: sort columns (after first two) by year
    cols = ['LA Code', 'LA Name'] + sorted(pivoted.columns[2:])
    pivoted = pivoted[cols]

    pivoted
    return (pivoted,)


@app.cell
def _(pivoted):
    pivoted.to_csv('processed_data/house_prices_formatted.csv')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
