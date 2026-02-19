"""Sales pricing preparation pipeline for monthly Customer x Product analysis.

This module ingests sales data from Excel, standardizes key columns, aggregates to
monthly product-customer grain, fills missing months, and computes pricing change
views (MoM, YoY, YTD vs PYTD) for downstream PVM analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


COLS: dict[str, Optional[str]] = {
    "global_id": "Global ID",
    "cust_nbr": "ROSS_CUST_NBR",
    "dollars": "DOLLARS (Invoiced Amt) USD",
    "pounds": "POUNDS (Invoice Qty) LB",
    "gl_date": "GL_Date",
    "sbu": None,
    "plant": None,
}


def _normalize_col_name(name: Any) -> str:
    """Normalize a column name by trimming and collapsing internal whitespace."""
    return " ".join(str(name).strip().split())


def _resolve_column_mapping(df: pd.DataFrame, col_map: dict[str, Optional[str]]) -> dict[str, str]:
    """Resolve canonical column names to existing DataFrame columns.

    Supports exact configured names and safe fallback matching against known aliases.
    """
    df_cols_norm_to_actual = {_normalize_col_name(c).lower(): c for c in df.columns}

    aliases: dict[str, list[str]] = {
        "global_id": ["global id", "global_id", "product id", "product identifier"],
        "cust_nbr": ["ross_cust_nbr", "ross cust nbr", "customer number", "customer"],
        "dollars": [
            "dollars (invoiced amt) usd",
            "invoiced amt usd",
            "invoice dollars",
            "dollars",
            "sales usd",
        ],
        "pounds": ["pounds (invoice qty) lb", "invoice qty lb", "pounds", "quantity lb"],
        "gl_date": ["gl_date", "gl date", "invoice date", "date"],
        "sbu": ["sbu", "business unit", "strategic business unit"],
        "plant": ["plant", "plant code", "manufacturing plant"],
    }

    resolved: dict[str, str] = {}

    for canonical, configured_name in col_map.items():
        if configured_name is None:
            continue

        configured_norm = _normalize_col_name(configured_name).lower()
        if configured_norm in df_cols_norm_to_actual:
            resolved[canonical] = df_cols_norm_to_actual[configured_norm]
            continue

        if canonical in aliases:
            found = next((a for a in aliases[canonical] if a in df_cols_norm_to_actual), None)
            if found is not None:
                resolved[canonical] = df_cols_norm_to_actual[found]
                print(
                    f"[WARN] Configured column '{configured_name}' not found for '{canonical}'. "
                    f"Using fallback '{resolved[canonical]}'."
                )
                continue

        raise KeyError(
            f"Required canonical field '{canonical}' could not be resolved from configured name "
            f"'{configured_name}'. Available columns: {list(df.columns)}"
        )

    required = ["global_id", "cust_nbr", "dollars", "pounds", "gl_date"]
    missing_required = [k for k in required if k not in resolved]
    if missing_required:
        raise KeyError(f"Missing required mapped columns: {missing_required}")

    return resolved


def read_sales_excel(
    excel_path: str | Path,
    col_map: dict[str, Optional[str]],
    sheet_name: Optional[str] = None,
    header: int = 0,
    dtype: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Read and minimally clean source sales Excel data.

    Parameters
    ----------
    excel_path:
        Path to source Excel file.
    col_map:
        Canonical-to-actual column mapping configuration.
    sheet_name:
        Optional worksheet name. If None, first sheet is used.
    header:
        Header row index.
    dtype:
        Optional dtype specification passed to pandas.read_excel.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header, dtype=dtype)
    df.columns = [_normalize_col_name(c) for c in df.columns]

    resolved = _resolve_column_mapping(df, col_map)

    rename_to_canonical = {
        resolved["global_id"]: "Global ID",
        resolved["cust_nbr"]: "ROSS_CUST_NBR",
        resolved["dollars"]: "DOLLARS (Invoiced Amt) USD",
        resolved["pounds"]: "POUNDS (Invoice Qty) LB",
        resolved["gl_date"]: "GL_Date",
    }

    if "sbu" in resolved:
        rename_to_canonical[resolved["sbu"]] = "SBU"
    if "plant" in resolved:
        rename_to_canonical[resolved["plant"]] = "Plant"

    df = df.rename(columns=rename_to_canonical)

    for num_col in ["DOLLARS (Invoiced Amt) USD", "POUNDS (Invoice Qty) LB"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

    invalid_date_before = df["GL_Date"].isna().sum()
    df["GL_Date"] = pd.to_datetime(df["GL_Date"], errors="coerce")
    invalid_date_after = df["GL_Date"].isna().sum()
    dropped_invalid = int(invalid_date_after)
    if dropped_invalid > 0:
        print(
            f"[WARN] Invalid GL_Date rows: {dropped_invalid} "
            f"(pre-existing nulls: {int(invalid_date_before)}). Dropping these rows."
        )
        df = df.dropna(subset=["GL_Date"]).copy()

    df["Global ID"] = df["Global ID"].astype(str).str.strip()
    df["ROSS_CUST_NBR"] = df["ROSS_CUST_NBR"].astype(str).str.strip()

    return df


def _mode_or_first(series: pd.Series, label: str, key: tuple[Any, Any, pd.Timestamp]) -> Any:
    """Return mode for group metadata, warning when non-unique values exist."""
    non_null = series.dropna()
    if non_null.empty:
        return np.nan

    unique_count = non_null.nunique(dropna=True)
    if unique_count > 1:
        print(f"[WARN] Non-unique {label} for key={key}. Choosing mode.")

    mode_vals = non_null.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else non_null.iloc[0]


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate invoice-level data to monthly Product x Customer grain."""
    work = df.copy()
    work["Month"] = work["GL_Date"].dt.to_period("M").dt.to_timestamp()

    group_cols = ["Global ID", "ROSS_CUST_NBR", "Month"]

    agg_dict: dict[str, Any] = {
        "DOLLARS (Invoiced Amt) USD": "sum",
        "POUNDS (Invoice Qty) LB": "sum",
    }

    if "SBU" in work.columns:
        agg_dict["SBU"] = lambda x: _mode_or_first(x, "SBU", (x.name if hasattr(x, "name") else None))
    if "Plant" in work.columns:
        agg_dict["Plant"] = lambda x: _mode_or_first(x, "Plant", (x.name if hasattr(x, "name") else None))

    monthly = work.groupby(group_cols, as_index=False).agg(agg_dict)
    monthly = monthly.rename(
        columns={
            "DOLLARS (Invoiced Amt) USD": "SumDollars_USD",
            "POUNDS (Invoice Qty) LB": "SumPounds_LB",
        }
    )

    monthly["Price_USD_per_LB"] = np.where(
        monthly["SumPounds_LB"] == 0,
        np.nan,
        monthly["SumDollars_USD"] / monthly["SumPounds_LB"],
    )

    return monthly


def _expand_months_per_group(group: pd.DataFrame) -> pd.DataFrame:
    """Expand one product-customer group to complete monthly range."""
    min_month = group["Month"].min()
    max_month = group["Month"].max()
    all_months = pd.date_range(min_month, max_month, freq="MS")

    expanded = group.set_index("Month").reindex(all_months)
    expanded.index.name = "Month"
    expanded = expanded.reset_index()

    expanded["Global ID"] = group["Global ID"].iloc[0]
    expanded["ROSS_CUST_NBR"] = group["ROSS_CUST_NBR"].iloc[0]

    if "SBU" in group.columns:
        expanded["SBU"] = _mode_or_first(group["SBU"], "SBU", (group["Global ID"].iloc[0], group["ROSS_CUST_NBR"].iloc[0], "all"))
    if "Plant" in group.columns:
        expanded["Plant"] = _mode_or_first(group["Plant"], "Plant", (group["Global ID"].iloc[0], group["ROSS_CUST_NBR"].iloc[0], "all"))

    return expanded


def build_complete_series(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build complete monthly rows and fill no-sales months with zero dollars/pounds."""
    key_cols = ["Global ID", "ROSS_CUST_NBR"]

    expanded = (
        monthly.sort_values(key_cols + ["Month"])
        .groupby(key_cols, group_keys=False)
        .apply(_expand_months_per_group)
        .reset_index(drop=True)
    )

    for c in ["SumDollars_USD", "SumPounds_LB"]:
        expanded[c] = expanded[c].fillna(0.0)

    expanded["Price_USD_per_LB"] = np.where(
        expanded["SumPounds_LB"] == 0,
        np.nan,
        expanded["SumDollars_USD"] / expanded["SumPounds_LB"],
    )

    expanded["Price_Filled_USD_per_LB"] = (
        expanded.groupby(key_cols)["Price_USD_per_LB"].transform(lambda s: s.ffill().bfill())
    )

    all_nan_series = (
        expanded.groupby(key_cols)["Price_USD_per_LB"].apply(lambda s: s.notna().sum() == 0).rename("AllNaN_PriceSeries")
    )

    expanded = expanded.merge(all_nan_series.reset_index(), on=key_cols, how="left")
    expanded.loc[expanded["AllNaN_PriceSeries"], "Price_Filled_USD_per_LB"] = np.nan

    dup_count = expanded.duplicated(subset=["Global ID", "ROSS_CUST_NBR", "Month"]).sum()
    assert dup_count == 0, f"Found {dup_count} duplicate keys at final grain."

    return expanded


def add_pricing_views(pricing_monthly: pd.DataFrame) -> pd.DataFrame:
    """Add MoM, YoY, and YTD vs PYTD pricing change metrics."""
    key_cols = ["Global ID", "ROSS_CUST_NBR"]
    out = pricing_monthly.sort_values(key_cols + ["Month"]).copy()

    grp = out.groupby(key_cols)

    out["Prev_Month_Price"] = grp["Price_Filled_USD_per_LB"].shift(1)
    out["MoM_Abs_Change"] = out["Price_Filled_USD_per_LB"] - out["Prev_Month_Price"]
    out["MoM_Pct_Change"] = np.where(
        out["Prev_Month_Price"].isin([0]) | out["Prev_Month_Price"].isna(),
        np.nan,
        out["MoM_Abs_Change"] / out["Prev_Month_Price"],
    )

    out["Prev_Year_Month_Price"] = grp["Price_Filled_USD_per_LB"].shift(12)
    out["YoY_Abs_Change"] = out["Price_Filled_USD_per_LB"] - out["Prev_Year_Month_Price"]
    out["YoY_Pct_Change"] = np.where(
        out["Prev_Year_Month_Price"].isin([0]) | out["Prev_Year_Month_Price"].isna(),
        np.nan,
        out["YoY_Abs_Change"] / out["Prev_Year_Month_Price"],
    )

    out["Year"] = out["Month"].dt.year
    out["MonthNum"] = out["Month"].dt.month

    out["YTD_Dollars"] = grp["SumDollars_USD"].cumsum()
    out["YTD_Pounds"] = grp["SumPounds_LB"].cumsum()
    out["YTD_Avg_Price"] = np.where(
        out["YTD_Pounds"] == 0,
        np.nan,
        out["YTD_Dollars"] / out["YTD_Pounds"],
    )

    out["PYTD_Avg_Price"] = grp["YTD_Avg_Price"].shift(12)
    out["YTD_Abs_Change"] = out["YTD_Avg_Price"] - out["PYTD_Avg_Price"]
    out["YTD_Pct_Change"] = np.where(
        out["PYTD_Avg_Price"].isin([0]) | out["PYTD_Avg_Price"].isna(),
        np.nan,
        out["YTD_Abs_Change"] / out["PYTD_Avg_Price"],
    )

    return out


def print_quality_summary(pricing_monthly: pd.DataFrame) -> None:
    """Print quality checks and summary statistics for the monthly pricing table."""
    n_customers = pricing_monthly["ROSS_CUST_NBR"].nunique(dropna=True)
    n_products = pricing_monthly["Global ID"].nunique(dropna=True)
    n_pairs = pricing_monthly[["Global ID", "ROSS_CUST_NBR"]].drop_duplicates().shape[0]
    n_rows = len(pricing_monthly)
    n_months = pricing_monthly["Month"].nunique(dropna=True)

    pct_no_sales = (pricing_monthly["SumPounds_LB"] == 0).mean() * 100
    pct_missing_price = pricing_monthly["Price_USD_per_LB"].isna().mean() * 100
    pct_missing_filled = pricing_monthly["Price_Filled_USD_per_LB"].isna().mean() * 100
    all_nan_series_count = int(pricing_monthly["AllNaN_PriceSeries"].sum())

    print("\n=== Pricing Monthly Quality Summary ===")
    print(f"Unique customers: {n_customers:,}")
    print(f"Unique products: {n_products:,}")
    print(f"Unique customer-product pairs: {n_pairs:,}")
    print(f"Rows (total months): {n_rows:,}")
    print(f"Distinct Month values: {n_months:,}")
    print(f"% months with no sales (SumPounds_LB == 0): {pct_no_sales:.2f}%")
    print(f"% missing Price_USD_per_LB before fill: {pct_missing_price:.2f}%")
    print(f"% missing Price_Filled_USD_per_LB after fill: {pct_missing_filled:.2f}%")
    print(f"Series with all-NaN filled price: {all_nan_series_count:,}")


def run_pricing_pipeline(
    excel_path: str | Path,
    col_map: dict[str, Optional[str]],
    sheet_name: Optional[str] = None,
    header: int = 0,
    dtype: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run end-to-end pricing prep pipeline.

    Returns
    -------
    pricing_monthly:
        Base monthly pricing table at Product x Customer x Month grain.
    pricing_views:
        pricing_monthly plus MoM, YoY, YTD vs PYTD derived columns.
    """
    df = read_sales_excel(
        excel_path=excel_path,
        col_map=col_map,
        sheet_name=sheet_name,
        header=header,
        dtype=dtype,
    )

    monthly = aggregate_monthly(df)
    pricing_monthly = build_complete_series(monthly)

    base_cols = [
        "Global ID",
        "ROSS_CUST_NBR",
        "Month",
        "SumDollars_USD",
        "SumPounds_LB",
        "Price_USD_per_LB",
        "Price_Filled_USD_per_LB",
    ]

    optional_cols = [c for c in ["SBU", "Plant", "AllNaN_PriceSeries"] if c in pricing_monthly.columns]
    pricing_monthly = pricing_monthly[base_cols + optional_cols].sort_values(
        ["Global ID", "ROSS_CUST_NBR", "Month"]
    )

    pricing_views = add_pricing_views(pricing_monthly)

    print_quality_summary(pricing_monthly)

    return pricing_monthly, pricing_views


if __name__ == "__main__":
    # Example usage
    excel_path = Path("/path/to/your/sales_file.xlsx")
    sheet_name: Optional[str] = None

    pricing_monthly, pricing_views = run_pricing_pipeline(
        excel_path=excel_path,
        col_map=COLS,
        sheet_name=sheet_name,
        header=0,
        dtype=None,
    )

    print("\npricing_monthly preview:")
    print(pricing_monthly.head())
    print("\npricing_views preview:")
    print(pricing_views.head())
