from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


def frequency_table(series: pd.Series) -> pd.DataFrame:
    valid = series.dropna()
    counts = valid.value_counts()
    total = int(counts.sum())

    result = counts.rename_axis("Κατηγορία").reset_index(name="Απαντήσεις")
    result["Ποσοστό"] = (
        result["Απαντήσεις"] / total * 100
    ).round(1) if total else 0.0

    return result


def preference_summary(series: pd.Series) -> dict[str, Any]:
    table = frequency_table(series)

    if table.empty:
        return {
            "winner": "Δεν υπάρχουν δεδομένα",
            "winner_count": 0,
            "winner_percentage": 0.0,
            "table": table,
        }

    winner = table.iloc[0]

    return {
        "winner": str(winner["Κατηγορία"]),
        "winner_count": int(winner["Απαντήσεις"]),
        "winner_percentage": float(winner["Ποσοστό"]),
        "table": table,
    }


def association_test(
    dataframe: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> dict[str, Any]:
    data = dataframe[[row_column, column_column]].dropna()
    contingency = pd.crosstab(data[row_column], data[column_column])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "valid": False,
            "message": "Δεν υπάρχουν αρκετές κατηγορίες για έλεγχο.",
            "table": contingency,
        }

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    expected_array = np.asarray(expected)
    use_fisher = contingency.shape == (2, 2) and (expected_array < 5).any()

    if use_fisher:
        odds_ratio, p_value = fisher_exact(contingency.to_numpy())
        test_name = "Fisher's Exact Test"
        statistic = float(odds_ratio)
    else:
        test_name = "Chi-square"
        statistic = float(chi2)

    n = int(contingency.to_numpy().sum())
    minimum_dimension = min(contingency.shape) - 1
    cramers_v = (
        sqrt(chi2 / (n * minimum_dimension))
        if n > 0 and minimum_dimension > 0
        else 0.0
    )

    return {
        "valid": True,
        "test": test_name,
        "statistic": statistic,
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "significant": bool(p_value < 0.05),
        "table": contingency,
    }


def automatic_insight(
    series: pd.Series,
    label: str,
) -> str:
    summary = preference_summary(series)

    if summary["winner_count"] == 0:
        return f"Δεν υπάρχουν διαθέσιμα δεδομένα για {label}."

    return (
        f"Η συχνότερη επιλογή για {label} ήταν «{summary['winner']}» "
        f"με {summary['winner_count']} απαντήσεις "
        f"({summary['winner_percentage']:.1f}%)."
    )
