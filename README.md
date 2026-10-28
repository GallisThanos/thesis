# thesis

Two Python/Streamlit applications supporting a thesis on computational analysis of music mixes and listening-test evaluation.

- **Feature extraction:** upload reference WAV files and a mix, extract loudness, spectral, stereo and temporal features, compare against the reference profile, and generate deterministic rule-based observations. Optional Gemini output rephrases these observations; generated wording is not guaranteed deterministic.
- **Questionnaire dashboard:** upload the questionnaire CSV, filter participants, inspect preferences across three comparisons, and explore associations using Chi-square/Fisher tests and Cramér’s V.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. From the project directory:

```bash
uv sync --locked
```

Python 3.12 is pinned in `.python-version`; the project supports Python 3.12–3.13. `uv` creates `.venv` and installs the project packages and dependencies from `uv.lock`. No manual `PYTHONPATH` or virtual-environment activation is needed.

## Run

```bash
# Audio analysis
uv run streamlit run src/feature_extraction/app.py

# Questionnaire analysis (use another terminal/port to run both)
uv run streamlit run src/questionnaire_dashboard/app.py --server.port 8502

# Batch reference profiles: one subdirectory per genre
uv run python -m feature_extraction.main /path/to/dataset
```

Open the local URL printed by Streamlit. The applications remain separate entry points within one repository and environment.

### Optional Gemini

Core analysis works without an API key; AI is off by default and unavailable without a key. Set `GOOGLE_API_KEY` in your environment before launching the audio app. Optionally override `GEMINI_MODEL` (default `gemini-2.5-flash`).

PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-key"
uv run streamlit run src/feature_extraction/app.py
```

Bash:

```bash
export GOOGLE_API_KEY="your-key"
uv run streamlit run src/feature_extraction/app.py
```

Alternatively copy `.env.example` to `.env`, fill in the key and explicitly load it with `uv run --env-file .env streamlit run src/feature_extraction/app.py`. Never commit credentials. When enabled, Gemini receives the feature summaries and interpretation. The existing `google-generativeai` integration is retained; SDK migration is a separate follow-up.

## Questionnaire data and results

Use the original questionnaire export (CSV). The existing preprocessing maps questionnaire column headings and answers; arbitrary survey schemas are not supported. No actual responses were supplied with this migration.

- `data/private/`: local raw responses; ignored by Git.
- `data/results/`: reviewed, anonymized or aggregate results suitable for version control.

Do not publish participant names, email addresses or identifying free-text answers. Preserve raw data separately with a private backup; `.gitignore` is not a backup. CSV uploads are read in memory by the questionnaire app. Audio analysis creates temporary WAV files; existing cleanup is not guaranteed after exceptions. Run locally for research use: authentication and production deployment are outside this migration. Streamlit's default upload limit applies (200 MB per file).

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run ruff format .
uv run mypy src
uv run pre-commit install
uv run pip-audit
```

Update a dependency with `uv add package-name`, or intentionally refresh compatible locked versions with `uv lock --upgrade`. Commit both `pyproject.toml` and `uv.lock`.

GitHub Actions defines lint, formatting and typing checks on Python 3.12/3.13, plus a non-blocking dependency audit. Dependabot configuration schedules weekly dependency updates. These run after the repository is pushed to GitHub; branch protection must be enabled in repository settings, requiring the CI checks.

## Structure and provenance

```text
src/feature_extraction/       # audio analysis and its Streamlit UI
src/questionnaire_dashboard/  # CSV ingestion, preprocessing, statistics and UI
data/results/                # shareable study results
pyproject.toml                # package, dependencies and tool settings
uv.lock                       # resolved dependency versions
```

Merged from [Feature-Extraction-Second-Version-Thesis](https://github.com/GallisThanos/Feature-Extraction-Second-Version-Thesis) and [Questionnaire_Analysis_Dashboard](https://github.com/GallisThanos/Questionnaire_Analysis_Dashboard). Both source histories are retained in the accompanying Git bundle. The local `statistics.py` was renamed `stats.py` to avoid shadowing Python's standard library. Scientific feature calculations and questionnaire test-selection logic are unchanged; this migration is packaging and maintenance work, not a validation of the research methodology.

## Create the GitHub repository

Create an empty repository named `thesis` on GitHub (without initializing a README). If restoring from the accompanying bundle:

```bash
git clone thesis.bundle thesis
cd thesis
git remote remove origin
git remote add origin https://github.com/GallisThanos/thesis.git
git push -u origin main
```

The original two repositories remain available and unchanged.
