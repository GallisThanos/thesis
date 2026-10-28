# Study results

- `responses_FINAL.csv`: the final questionnaire export from AidaForm (51 responses, July 2026, plus one earlier response from May 2026 that contains only the second question block). It is the unmodified export, so it can be uploaded directly to the questionnaire dashboard:

  ```bash
  uv run streamlit run src/questionnaire_dashboard/app.py
  ```

The file contains only multiple-choice answers: no names, emails or free text. The `Link` column holds AidaForm response URLs, which are only reachable by the form owner. Keep any identifiable raw data in `data/private/` (ignored by Git) and maintain a separate private backup.
