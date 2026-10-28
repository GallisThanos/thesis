from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


class QuestionnairePreprocessor:
    TECHNICAL_COLUMNS = {"#", "Created", "Link"}

    EXPERIENCE_ORDER = [
        "Δεν έχω κάποια σχετική ενασχόληση",
        "Έχω βασικές γνώσεις / ερασιτεχνική ενασχόληση",
        "Έχω προχωρημένες γνώσεις και συστηματική ενασχόληση",
        "Επαγγελματικό επίπεδο / εργάζομαι στον χώρο του ήχου ή της μουσικής παραγωγής",
    ]

    DIFFERENCE_ORDER = [
        "Ελάχιστη",
        "Μέτρια",
        "Σημαντική",
        "Πολύ σημαντική",
    ]

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.raw = dataframe.copy()
        self.dataframe: pd.DataFrame | None = None
        self.detected_blocks = 0

    def process(self) -> pd.DataFrame:
        dataframe = self.raw.copy()
        dataframe.columns = [str(column).strip() for column in dataframe.columns]

        question_columns = [
            column for column in dataframe.columns
            if column not in self.TECHNICAL_COLUMNS
        ]

        blocks = self._split_question_blocks(question_columns)
        self.detected_blocks = len(blocks)

        output = pd.DataFrame(index=dataframe.index)

        if "Created" in dataframe.columns:
            output["created"] = pd.to_datetime(dataframe["Created"], errors="coerce")

        role_candidates: list[pd.Series] = []
        experience_candidates: list[pd.Series] = []
        comparison_candidates: dict[str, list[pd.Series]] = {
            f"comparison_{comparison}_{field}": []
            for comparison in range(1, 4)
            for field in ("preference", "difference", "quality", "factor")
        }

        for block in blocks:
            semantic_columns = self._classify_block(block)

            if semantic_columns.get("sound_role"):
                role_candidates.append(dataframe[semantic_columns["sound_role"]])

            if semantic_columns.get("experience_level"):
                experience_candidates.append(
                    dataframe[semantic_columns["experience_level"]]
                )

            for target, source_column in semantic_columns.items():
                if target.startswith("comparison_") and source_column:
                    comparison_candidates[target].append(dataframe[source_column])

        output["sound_role"] = self._coalesce(role_candidates)
        output["experience_level"] = self._coalesce(experience_candidates)

        for target, candidates in comparison_candidates.items():
            output[target] = self._coalesce(candidates)

        text_columns = output.select_dtypes(include="object").columns
        for column in text_columns:
            output[column] = output[column].apply(self._clean_text)

        required = [
            "sound_role",
            "experience_level",
            *comparison_candidates.keys(),
        ]

        missing_columns = [
            column for column in required
            if column not in output.columns or output[column].isna().all()
        ]

        if missing_columns:
            raise ValueError(
                "Δεν αναγνωρίστηκαν σωστά οι ακόλουθες ερωτήσεις: "
                + ", ".join(missing_columns)
            )

        output = output.dropna(how="all", subset=required)
        output = output.drop_duplicates(subset=required).reset_index(drop=True)

        experience_scores = {
            value: index + 1
            for index, value in enumerate(self.EXPERIENCE_ORDER)
        }
        difference_scores = {
            value: index + 1
            for index, value in enumerate(self.DIFFERENCE_ORDER)
        }

        output["experience_score"] = output["experience_level"].map(
            experience_scores
        )

        for comparison in range(1, 4):
            difference_column = f"comparison_{comparison}_difference"
            output[f"comparison_{comparison}_difference_score"] = (
                output[difference_column].map(difference_scores)
            )

        self.dataframe = output
        return output.copy()

    def summary(self) -> dict[str, int]:
        if self.dataframe is None:
            raise ValueError("Πρέπει πρώτα να εκτελεστεί το process().")

        return {
            "responses": len(self.dataframe),
            "demographic_fields": 2,
            "comparisons": 3,
            "questions_per_comparison": 4,
            "source_blocks": self.detected_blocks,
        }

    @staticmethod
    def comparison_columns(comparison: int) -> dict[str, str]:
        if comparison not in (1, 2, 3):
            raise ValueError("Η σύγκριση πρέπει να είναι 1, 2 ή 3.")

        return {
            field: f"comparison_{comparison}_{field}"
            for field in ("preference", "difference", "quality", "factor")
        } | {
            "difference_score": f"comparison_{comparison}_difference_score"
        }

    def _split_question_blocks(self, columns: list[str]) -> list[list[str]]:
        blocks: list[list[str]] = []
        current: list[str] = []
        previous_number: int | None = None

        for column in columns:
            number = self._question_number(column)

            if (
                current
                and number is not None
                and previous_number is not None
                and number <= previous_number
            ):
                blocks.append(current)
                current = []

            current.append(column)

            if number is not None:
                previous_number = number

        if current:
            blocks.append(current)

        return blocks

    def _classify_block(self, block: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        grouped: dict[str, list[str]] = {
            "preference": [],
            "difference": [],
            "quality": [],
            "factor": [],
        }

        for column in block:
            normalized = self._normalize(column)

            if "σχεση σας με τον ηχο" in normalized:
                result["sound_role"] = column
            elif "επιπεδο γνωσεων και εμπειριας" in normalized:
                result["experience_level"] = column
            elif "ποια εκδοχη προτιματε" in normalized:
                grouped["preference"].append(column)
            elif "ποσο σημαντικ" in normalized and "διαφορα" in normalized:
                grouped["difference"].append(column)
            elif "καλυτερη συνολικη ποιοτητα" in normalized:
                grouped["quality"].append(column)
            elif "τι επηρεασε περισσοτερο" in normalized:
                grouped["factor"].append(column)

        for field, columns in grouped.items():
            for index, column in enumerate(columns[:3], start=1):
                result[f"comparison_{index}_{field}"] = column

        return result

    @staticmethod
    def _coalesce(series_list: list[pd.Series]) -> pd.Series:
        if not series_list:
            return pd.Series(dtype="object")

        result = series_list[0].copy()

        for series in series_list[1:]:
            blank = result.isna() | result.astype("string").str.strip().eq("")
            result = result.where(~blank, series)

        return result

    @staticmethod
    def _question_number(column: str) -> int | None:
        match = re.match(r"^\s*(\d+)\.", column)
        return int(match.group(1)) if match else None

    @staticmethod
    def _normalize(value: Any) -> str:
        text = unicodedata.normalize("NFD", str(value).lower())
        text = "".join(
            character for character in text
            if unicodedata.category(character) != "Mn"
        )
        text = re.sub(r"^\s*\d+\.\s*", "", text)
        text = re.sub(r"[^a-zα-ω0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_text(value: Any) -> Any:
        if pd.isna(value):
            return value

        text = unicodedata.normalize("NFC", str(value)).strip()
        text = re.sub(r"\s+", " ", text)

        replacements = {
            "'Ενταση": "Ένταση",
            "Ενταση": "Ένταση",
            "Εκδοχή Πρώτη": "Εκδοχή πρώτη",
            "Εκδοχή Δεύτερη": "Εκδοχή δεύτερη",
        }

        return replacements.get(text, text)
