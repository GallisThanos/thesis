from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


class CSVLoader:
    ENCODINGS = ("utf-8-sig", "utf-8", "cp1253", "windows-1252", "latin-1")
    SEPARATORS = (",", ";", "\t")

    def __init__(self) -> None:
        self.dataframe: pd.DataFrame | None = None
        self.filename: str | None = None
        self.encoding: str | None = None
        self.separator: str | None = None
        self.platform = "Unknown"

    def load_csv(self, source: Any) -> pd.DataFrame:
        self._reset()
        self.filename = self._filename(source)
        raw = self._bytes(source)

        if not raw:
            raise ValueError("Το αρχείο είναι κενό.")

        for encoding in self.ENCODINGS:
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue

            for separator in self.SEPARATORS:
                try:
                    dataframe = pd.read_csv(StringIO(text), sep=separator)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    continue

                if len(dataframe.columns) <= 1:
                    continue

                self.dataframe = dataframe
                self.encoding = encoding
                self.separator = separator
                self.platform = self.detect_platform()
                return dataframe.copy()

        raise ValueError("Δεν ήταν δυνατή η ανάγνωση του CSV.")

    def detect_platform(self) -> str:
        if self.dataframe is None:
            return "Unknown"

        columns = {str(column).strip().lower() for column in self.dataframe.columns}

        if {"#", "created", "link"}.issubset(columns):
            return "AidaForm"
        if "timestamp" in columns or "χρονοσήμανση" in columns:
            return "Google Forms"
        if {"start time", "completion time"}.issubset(columns):
            return "Microsoft Forms"

        return "Unknown"

    def validate(self) -> dict[str, Any]:
        if self.dataframe is None:
            return {"valid": False, "message": "Δεν έχει φορτωθεί CSV."}

        if self.dataframe.empty:
            return {"valid": False, "message": "Το CSV δεν περιέχει απαντήσεις."}

        return {
            "valid": True,
            "message": "Το CSV φορτώθηκε επιτυχώς.",
            "rows": len(self.dataframe),
            "columns": len(self.dataframe.columns),
            "platform": self.platform,
            "filename": self.filename,
            "encoding": self.encoding,
            "separator": {",": "comma", ";": "semicolon", "\t": "tab"}.get(self.separator or ""),
        }

    def _bytes(self, source: Any) -> bytes:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Το αρχείο δεν βρέθηκε: {path}")
            return path.read_bytes()

        if hasattr(source, "getvalue"):
            value = source.getvalue()
        elif hasattr(source, "read"):
            value = source.read()
        else:
            raise TypeError("Μη υποστηριζόμενη πηγή αρχείου.")

        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, bytes):
            return value

        raise TypeError("Μη υποστηριζόμενο περιεχόμενο αρχείου.")

    @staticmethod
    def _filename(source: Any) -> str | None:
        if isinstance(source, (str, Path)):
            return Path(source).name
        name = getattr(source, "name", None)
        return Path(str(name)).name if name else None

    def _reset(self) -> None:
        self.dataframe = None
        self.filename = None
        self.encoding = None
        self.separator = None
        self.platform = "Unknown"
