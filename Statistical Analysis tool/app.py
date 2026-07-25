import altair as alt
import pandas as pd
import streamlit as st

from loader import CSVLoader
from preprocessing import QuestionnairePreprocessor
from statistics import association_test, frequency_table, preference_summary


st.set_page_config(
    page_title="Questionnaire Results Evaluation",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ACCENT = "#6C63FF"
ACCENT_SOFT = "#8B86FF"
TEXT = "#172033"
MUTED = "#667085"
GRID = "#E8EAF0"
SUCCESS = "#1F9D73"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: "Inter", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 8% 0%, rgba(108, 99, 255, 0.10), transparent 28rem),
                radial-gradient(circle at 92% 8%, rgba(54, 211, 153, 0.07), transparent 24rem),
                #F7F8FC;
        }}

        .block-container {{
            max-width: 1200px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4 {{
            color: {TEXT};
            letter-spacing: -0.03em;
        }}

        .hero {{
            padding: 1.8rem 1.6rem;
            margin-bottom: 1.2rem;
            border-radius: 22px;
            color: white;
            text-align: center;
            background: linear-gradient(135deg, #5E56E8 0%, #8178FF 58%, #9F98FF 100%);
            box-shadow: 0 18px 45px rgba(83, 75, 190, 0.22);
        }}

        .hero-title {{
            margin: 0;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 750;
            letter-spacing: -0.05em;
        }}

        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(108, 99, 255, 0.16);
            border-radius: 18px;
            padding: .4rem .7rem;
            box-shadow: 0 10px 25px rgba(34, 43, 69, 0.06);
        }}

        [data-testid="stMetric"] {{
            min-height: 126px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(23, 32, 51, 0.07);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 10px 28px rgba(34, 43, 69, 0.06);
        }}

        [data-testid="stMetricLabel"] {{
            color: {MUTED};
            font-size: .82rem;
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            color: {TEXT};
            font-size: 1.55rem;
            font-weight: 700;
        }}

        [data-testid="stMetricDelta"] {{
            font-size: .82rem;
        }}

        div[data-testid="stTabs"] button {{
            min-height: 46px;
            padding: 0 1rem;
            color: {MUTED};
            font-weight: 600;
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: {ACCENT};
        }}

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: {ACCENT};
        }}

        div[data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(23, 32, 51, 0.07);
            border-radius: 15px;
            box-shadow: 0 6px 18px rgba(34, 43, 69, 0.04);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {{
            border-radius: 12px;
        }}

        .section-card {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(23, 32, 51, 0.07);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 28px rgba(34, 43, 69, 0.05);
        }}

        .section-title {{
            color: {TEXT};
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: .25rem;
        }}

        .section-copy {{
            color: {MUTED};
            font-size: .88rem;
            line-height: 1.55;
        }}

        .result-box {{
            padding: 1rem 1.15rem;
            border-radius: 15px;
            border-left: 5px solid {ACCENT};
            background: rgba(108, 99, 255, 0.08);
            color: {TEXT};
            line-height: 1.6;
        }}

        .result-box.positive {{
            border-left-color: {SUCCESS};
            background: rgba(31, 157, 115, 0.09);
        }}

        .eyebrow {{
            color: {ACCENT};
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .10em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }}

        .footer-note {{
            color: {MUTED};
            font-size: .82rem;
            text-align: center;
            margin-top: 1rem;
        }}

        .stDownloadButton > button {{
            border: 0;
            border-radius: 12px;
            color: white;
            background: linear-gradient(135deg, #5E56E8, #8178FF);
            font-weight: 650;
            padding: .6rem 1rem;
        }}

        .stDownloadButton > button:hover {{
            color: white;
            border: 0;
            box-shadow: 0 8px 20px rgba(83, 75, 190, .22);
        }}

        @media (max-width: 760px) {{
            .block-container {{
                padding-top: 1rem;
            }}
            .hero {{
                padding: 1.25rem;
                border-radius: 18px;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def horizontal_bar_chart(
    table: pd.DataFrame,
    value_column: str = "Απαντήσεις",
    height: int = 300,
) -> alt.Chart:
    chart_data = table.copy()
    chart_data["Ετικέτα"] = chart_data["Κατηγορία"].astype(str)

    return (
        alt.Chart(chart_data)
        .mark_bar(
            cornerRadiusTopRight=7,
            cornerRadiusBottomRight=7,
            size=22,
        )
        .encode(
            x=alt.X(
                f"{value_column}:Q",
                title=None,
                axis=alt.Axis(grid=True, gridColor=GRID, tickCount=5),
            ),
            y=alt.Y(
                "Ετικέτα:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=270, labelPadding=10),
            ),
            color=alt.value(ACCENT),
            tooltip=[
                alt.Tooltip("Κατηγορία:N", title="Κατηγορία"),
                alt.Tooltip(f"{value_column}:Q", title=value_column),
                alt.Tooltip("Ποσοστό:Q", title="Ποσοστό", format=".1f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            domain=False,
            ticks=False,
        )
    )


def overview_chart(dataframe: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(dataframe)
        .mark_bar(
            cornerRadiusTopLeft=9,
            cornerRadiusTopRight=9,
            size=52,
        )
        .encode(
            x=alt.X("Σύγκριση:N", title=None, sort=None),
            y=alt.Y(
                "Ποσοστό:Q",
                title="Ποσοστό κυρίαρχης επιλογής (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(grid=True, gridColor=GRID),
            ),
            color=alt.Color(
                "Σύγκριση:N",
                legend=None,
                scale=alt.Scale(
                    range=[ACCENT, ACCENT_SOFT, "#B0ABFF"]
                ),
            ),
            tooltip=[
                alt.Tooltip("Σύγκριση:N"),
                alt.Tooltip("Κυρίαρχη επιλογή:N"),
                alt.Tooltip("Ποσοστό:Q", format=".1f"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            domain=False,
            ticks=False,
        )
    )


st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Questionnaire Results Evaluation</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Ανέβασε το αρχείο responses.csv",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.stop()

try:
    loader = CSVLoader()
    raw_dataframe = loader.load_csv(uploaded_file)
    validation = loader.validate()

    if not validation["valid"]:
        st.error(validation["message"])
        st.stop()

    preprocessor = QuestionnairePreprocessor(raw_dataframe)
    dataframe = preprocessor.process()
except Exception as error:
    st.error("Δεν ήταν δυνατή η επεξεργασία του αρχείου.")
    st.exception(error)
    st.stop()

with st.expander("Φίλτρα συμμετεχόντων", expanded=False):
    filter_col1, filter_col2 = st.columns(2)

    role_options = dataframe["sound_role"].dropna().unique().tolist()
    experience_options = dataframe["experience_level"].dropna().unique().tolist()

    selected_roles = filter_col1.multiselect(
        "Σχέση με τον ήχο",
        options=role_options,
        default=role_options,
    )

    selected_experience = filter_col2.multiselect(
        "Επίπεδο εμπειρίας",
        options=experience_options,
        default=experience_options,
    )

filtered = dataframe[
    dataframe["sound_role"].isin(selected_roles)
    & dataframe["experience_level"].isin(selected_experience)
].copy()

if filtered.empty:
    st.warning("Τα επιλεγμένα φίλτρα δεν επιστρέφουν απαντήσεις.")
    st.stop()

metric_col1, metric_spacer = st.columns([1, 3])
metric_col1.metric("Συμμετέχοντες", len(filtered))

st.write("")

overview_tab, participants_tab, results_tab, stats_tab = st.tabs(
    ["Επισκόπηση", "Συμμετέχοντες", "Αποτελέσματα", "Στατιστικά"]
)

with overview_tab:
    st.markdown("### Συνολική εικόνα")
    st.caption("Η κυρίαρχη επιλογή και το ποσοστό της σε κάθε μουσική σύγκριση.")

    overview_rows = []
    for comparison in range(1, 4):
        result = preference_summary(
            filtered[f"comparison_{comparison}_preference"]
        )
        overview_rows.append(
            {
                "Σύγκριση": f"Σύγκριση {comparison}",
                "Κυρίαρχη επιλογή": result["winner"],
                "Ποσοστό": result["winner_percentage"],
            }
        )

    overview = pd.DataFrame(overview_rows)

    card1, card2, card3 = st.columns(3, gap="medium")
    for card, row in zip((card1, card2, card3), overview_rows):
        with card:
            st.metric(
                label=row["Σύγκριση"],
                value=row["Κυρίαρχη επιλογή"],
                delta=f"{row['Ποσοστό']:.1f}% των απαντήσεων",
            )

    st.altair_chart(
        overview_chart(overview),
        use_container_width=True,
    )

with participants_tab:
    st.markdown("### Προφίλ συμμετεχόντων")
    st.caption("Κατανομή των συμμετεχόντων ανά σχέση με τον ήχο και εμπειρία.")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow">Δημογραφικά</div>
                <div class="section-title">Σχέση με τον ήχο</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        role_table = frequency_table(filtered["sound_role"])
        st.altair_chart(
            horizontal_bar_chart(role_table, height=330),
            use_container_width=True,
        )
        with st.expander("Προβολή αριθμητικών τιμών"):
            st.dataframe(role_table, use_container_width=True, hide_index=True)

    with right:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow">Δημογραφικά</div>
                <div class="section-title">Επίπεδο εμπειρίας</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        experience_table = frequency_table(filtered["experience_level"])
        st.altair_chart(
            horizontal_bar_chart(experience_table, height=330),
            use_container_width=True,
        )
        with st.expander("Προβολή αριθμητικών τιμών"):
            st.dataframe(
                experience_table,
                use_container_width=True,
                hide_index=True,
            )

with results_tab:
    top_col, spacer = st.columns([1, 2])
    comparison = top_col.selectbox(
        "Επιλογή σύγκρισης",
        options=[1, 2, 3],
        format_func=lambda value: f"Σύγκριση {value}",
    )

    columns = preprocessor.comparison_columns(comparison)
    preference = preference_summary(filtered[columns["preference"]])
    quality = preference_summary(filtered[columns["quality"]])
    mean_difference = filtered[columns["difference_score"]].mean()

    st.markdown(f"### Αποτελέσματα σύγκρισης {comparison}")
    st.caption("Σύνοψη των τεσσάρων βασικών ερωτήσεων της επιλεγμένης σύγκρισης.")

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric(
        "Προτιμώμενη εκδοχή",
        preference["winner"],
        f"{preference['winner_percentage']:.1f}%",
    )
    result_col2.metric(
        "Καλύτερη ποιότητα",
        quality["winner"],
        f"{quality['winner_percentage']:.1f}%",
    )
    result_col3.metric(
        "Μέση αντιληπτή διαφορά",
        f"{mean_difference:.2f} / 4",
    )

    preference_table = frequency_table(filtered[columns["preference"]])
    quality_table = frequency_table(filtered[columns["quality"]])
    difference_table = frequency_table(filtered[columns["difference"]])
    factor_table = frequency_table(filtered[columns["factor"]])

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown("#### Προτίμηση εκδοχής")
        st.altair_chart(
            horizontal_bar_chart(preference_table, height=240),
            use_container_width=True,
        )

        st.markdown("#### Σημαντικότητα διαφοράς")
        st.altair_chart(
            horizontal_bar_chart(difference_table, height=260),
            use_container_width=True,
        )

    with chart_col2:
        st.markdown("#### Καλύτερη συνολική ποιότητα")
        st.altair_chart(
            horizontal_bar_chart(quality_table, height=240),
            use_container_width=True,
        )

        st.markdown("#### Παράγοντας επιλογής")
        st.altair_chart(
            horizontal_bar_chart(factor_table, height=260),
            use_container_width=True,
        )

    with st.expander("Αναλυτικοί πίνακες"):
        detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
            ["Προτίμηση", "Ποιότητα", "Διαφορά", "Παράγοντας"]
        )
        detail_tab1.dataframe(
            preference_table, use_container_width=True, hide_index=True
        )
        detail_tab2.dataframe(
            quality_table, use_container_width=True, hide_index=True
        )
        detail_tab3.dataframe(
            difference_table, use_container_width=True, hide_index=True
        )
        detail_tab4.dataframe(
            factor_table, use_container_width=True, hide_index=True
        )

with stats_tab:
    st.markdown("### Στατιστική ερμηνεία")
    st.caption("Έλεγχος πιθανής σχέσης ανάμεσα στα χαρακτηριστικά των συμμετεχόντων και στις επιλογές τους.")

    with st.expander("Τι σημαίνουν οι δείκτες;", expanded=False):
        st.markdown(
            """
            **Chi-square:** ελέγχει αν δύο κατηγορικές μεταβλητές
            σχετίζονται μεταξύ τους.

            **p-value:** όταν είναι μικρότερο από **0,05**, υπάρχει
            στατιστικά σημαντική ένδειξη συσχέτισης.

            **Cramér's V:** δείχνει την ισχύ της σχέσης, από **0**
            (πολύ ασθενής) έως **1** (πολύ ισχυρή).
            """
        )

    control_col1, control_col2, control_col3 = st.columns(3)

    comparison = control_col1.selectbox(
        "Σύγκριση",
        [1, 2, 3],
        format_func=lambda value: f"Σύγκριση {value}",
        key="stats_comparison",
    )

    outcome = control_col2.selectbox(
        "Αποτέλεσμα",
        ["Προτίμηση", "Καλύτερη ποιότητα"],
    )

    demographic = control_col3.selectbox(
        "Χαρακτηριστικό συμμετεχόντων",
        ["Σχέση με τον ήχο", "Επίπεδο εμπειρίας"],
    )

    outcome_column = (
        f"comparison_{comparison}_preference"
        if outcome == "Προτίμηση"
        else f"comparison_{comparison}_quality"
    )
    demographic_column = (
        "sound_role"
        if demographic == "Σχέση με τον ήχο"
        else "experience_level"
    )

    result = association_test(
        filtered,
        demographic_column,
        outcome_column,
    )

    if not result["valid"]:
        st.warning(result["message"])
    else:
        cramers_v = result["cramers_v"]

        if cramers_v < 0.10:
            strength = "πολύ ασθενής"
        elif cramers_v < 0.30:
            strength = "ασθενής"
        elif cramers_v < 0.50:
            strength = "μέτρια"
        else:
            strength = "ισχυρή"

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Έλεγχος", result["test"])
        stat_col2.metric("p-value", f"{result['p_value']:.4f}")
        stat_col3.metric("Ισχύς σχέσης", strength.capitalize())

        st.markdown(
            f"""
            <div class="result-box">
                Το <strong>{result["test"]}</strong> εξετάζει αν το
                «{demographic}» σχετίζεται με το «{outcome}».
                Το p-value είναι <strong>{result["p_value"]:.4f}</strong>
                και το Cramér's V είναι <strong>{cramers_v:.3f}</strong>,
                επομένως η ισχύς της σχέσης χαρακτηρίζεται ως
                <strong>{strength}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        result_class = "positive" if result["significant"] else ""
        conclusion = (
            f"Υπάρχει στατιστικά σημαντική συσχέτιση μεταξύ του "
            f"«{demographic}» και του «{outcome}» στη σύγκριση {comparison}."
            if result["significant"]
            else
            f"Δεν υπάρχει στατιστικά σημαντική συσχέτιση μεταξύ του "
            f"«{demographic}» και του «{outcome}» στη σύγκριση {comparison}."
        )

        st.markdown(
            f"""
            <div class="result-box {result_class}">
                <strong>Συμπέρασμα:</strong> {conclusion}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Πίνακας αποτελεσμάτων"):
            st.dataframe(result["table"], use_container_width=True)

st.write("")
st.divider()

download_col, spacer = st.columns([1, 3])
cleaned_csv = dataframe.to_csv(index=False).encode("utf-8-sig")

with download_col:
    st.download_button(
        "Λήψη καθαρισμένων δεδομένων",
        data=cleaned_csv,
        file_name="cleaned_responses.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    '<p class="footer-note">Questionnaire Results Evaluation</p>',
    unsafe_allow_html=True,
)
