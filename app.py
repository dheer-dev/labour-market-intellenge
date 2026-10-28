from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Labour Market Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Labour Market Intelligence")
st.caption(
    "Explainable Multi-Source Labour Market Intelligence "
    "for Skill Gap Analysis and Career Decision Support"
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


FILES = {
    "demand": "occupation_skill_demand_analysis.csv",
    "model_dataset": "model_dataset.csv",
    "gaps": "occupation_skill_gap_analysis.csv",
    "priorities": "skill_gap_prioritisation.csv",
    "explanations": "skill_gap_explanations.csv",
    "career": "career_decision_support.csv",
    "ml_summary": "model_occupation_aware_grouped_cv_summary.csv",
    "baseline": "model_occupation_aware_baseline_summary.csv",
    "features": "model_feature_explanations.csv",
    "personal": "personal_skill_gaps.csv",
    "occupation_comparison": "user_occupation_skill_comparison.csv",
    "esco_relations": "esco_occupation_skill_relations.csv",
    "onet_occupations": "onet_occupations_clean.csv",
}

data = {key: load_csv(filename) for key, filename in FILES.items()}


def require_data(key):
    """Return a dataset or display an actionable error."""
    df = data.get(key)

    if df is None:
        st.error(
            f"Required file is missing: {FILES[key]}. "
            "Run the relevant data-processing script first."
        )
        return None

    return df


def format_number(value):
    return f"{value:,.0f}"


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Occupation Explorer",
        "Skill Gap Analysis",
        "Priority Recommendations",
        "ML Evaluation",
        "SHAP Explainability",
        "ML-Assisted Decision Support",
        "Personalised Skill Gaps",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Evidence sources: ESCO occupation-skill relationships "
    "and the available NYC job-posting snapshot."
)


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

if page == "Overview":
    st.header("Project Overview")

    demand = require_data("demand")
    gaps = require_data("gaps")
    priorities = require_data("priorities")
    esco = require_data("esco_relations")

    if all(df is not None for df in [demand, gaps, priorities, esco]):
        occupation_count = demand["occupation_label"].nunique()
        skill_count = demand["skill_label"].nunique()
        high_count = int((priorities["priority"] == "High").sum())
        moderate_count = int((priorities["priority"] == "Moderate").sum())
        emerging_count = int((priorities["priority"] == "Emerging").sum())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Analysed occupations", occupation_count)
        col2.metric("Market-demand records", len(demand))
        col3.metric("ESCO relationships", len(esco))
        col4.metric("Prioritised skill gaps", len(priorities))

        st.subheader("Skill-gap priority distribution")

        priority_counts = (
            priorities["priority"]
            .value_counts()
            .reindex(["High", "Moderate", "Emerging"], fill_value=0)
        )

        left, right = st.columns(2)

        with left:
            st.bar_chart(priority_counts)
            st.caption(
                "Priority bands use the project's existing prevalence-based rules."
            )

        with right:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Priority": ["High", "Moderate", "Emerging"],
                        "Number of gaps": [
                            high_count,
                            moderate_count,
                            emerging_count,
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.subheader("Occupation coverage")

        coverage = (
            demand.groupby("occupation_label")
            .agg(
                skills_observed=("skill_label", "nunique"),
                postings_with_skills=("postings_with_skills", "max"),
            )
            .reset_index()
            .sort_values("skills_observed", ascending=False)
        )

        st.dataframe(coverage, hide_index=True, width="stretch")

        st.info(
            "Interpretation: a market-only skill gap is a skill observed in the "
            "analysed job postings but not represented in the corresponding ESCO "
            "occupation-skill relationships. It is a data-supported gap candidate, "
            "not proof that ESCO is incorrect or that every worker needs the skill."
        )


# ---------------------------------------------------------
# Occupation Explorer
# ---------------------------------------------------------

elif page == "Occupation Explorer":
    st.header("Occupation Explorer")

    demand = require_data("demand")

    if demand is not None:
        occupations = sorted(demand["occupation_label"].dropna().unique())

        selected_occupation = st.selectbox(
            "Select an occupation",
            occupations,
        )

        occupation_data = demand[
            demand["occupation_label"] == selected_occupation
        ].copy()

        occupation_data = occupation_data.sort_values(
            "skill_posting_count",
            ascending=False,
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Skills in demand table", len(occupation_data))

        if not occupation_data.empty:
            col2.metric(
                "Postings with skills",
                format_number(occupation_data["postings_with_skills"].max()),
            )
            col3.metric(
                "Top skill prevalence",
                f"{occupation_data['skill_prevalence_percent'].max():.2f}%",
            )

        st.subheader("Most frequently observed skills")

        top_n = st.slider(
            "Number of skills to display",
            min_value=5,
            max_value=max(5, min(30, len(occupation_data))),
            value=min(10, max(5, len(occupation_data))),
        )

        chart_data = occupation_data.head(top_n).set_index("skill_label")

        st.bar_chart(chart_data["skill_prevalence_percent"])

        st.subheader("Skill-demand records")

        st.dataframe(
            occupation_data[
                [
                    "skill_label",
                    "skill_posting_count",
                    "skill_prevalence_percent",
                    "skill_rank",
                    "demand_category",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download occupation skill-demand CSV",
            data=occupation_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_occupation.replace(' ', '_')}_skill_demand.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# Skill Gap Analysis
# ---------------------------------------------------------

elif page == "Skill Gap Analysis":
    st.header("Occupation Skill Gap Analysis")

    gaps = require_data("gaps")

    if gaps is not None:
        occupations = sorted(gaps["occupation_label"].dropna().unique())

        selected_occupation = st.selectbox(
            "Select an occupation",
            occupations,
        )

        occupation_gaps = gaps[
            gaps["occupation_label"] == selected_occupation
        ].copy()

        category_options = sorted(
            occupation_gaps["gap_category"].dropna().unique()
        )

        selected_categories = st.multiselect(
            "Gap categories",
            options=category_options,
            default=category_options,
        )

        filtered = occupation_gaps[
            occupation_gaps["gap_category"].isin(selected_categories)
        ].copy()

        col1, col2, col3 = st.columns(3)

        col1.metric("Displayed records", len(filtered))
        col2.metric(
            "Market-only records",
            int((filtered["gap_category"] == "Market-only").sum()),
        )
        col3.metric(
            "ESCO-only records",
            int((filtered["gap_category"] == "ESCO-only").sum()),
        )

        st.caption(
            "Market-only: observed in the market data but absent from the "
            "corresponding ESCO occupation-skill relations. ESCO-only: present "
            "in ESCO but not observed in the analysed market data."
        )

        display_columns = [
            "skill_label",
            "gap_category",
            "market_skill_posting_count",
            "market_skill_prevalence_percent",
            "market_demand_category",
            "esco_relation_type",
            "esco_skill_type",
        ]

        st.dataframe(
            filtered[display_columns].sort_values(
                "market_skill_posting_count",
                ascending=False,
            ),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download filtered gap analysis",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_skill_gap_analysis.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# Priority Recommendations
# ---------------------------------------------------------

elif page == "Priority Recommendations":
    st.header("Priority Recommendations")

    explanations = require_data("explanations")

    if explanations is not None:
        occupations = sorted(
            explanations["occupation_label"].dropna().unique()
        )

        selected_occupation = st.selectbox(
            "Select an occupation",
            occupations,
        )

        priority_options = ["High", "Moderate", "Emerging"]

        selected_priorities = st.multiselect(
            "Priority levels",
            priority_options,
            default=priority_options,
        )

        recommendations = explanations[
            (explanations["occupation_label"] == selected_occupation)
            & (explanations["priority"].isin(selected_priorities))
        ].copy()

        recommendations = recommendations.sort_values(
            "market_skill_prevalence_percent",
            ascending=False,
        )

        st.subheader(f"Recommended skills: {selected_occupation}")

        if recommendations.empty:
            st.warning("No recommendations match the selected filters.")
        else:
            for _, row in recommendations.iterrows():
                title = (
                    f"{row['skill_label']} — {row['priority']} priority"
                )

                with st.expander(title):
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Posting count",
                        format_number(row["market_skill_posting_count"]),
                    )
                    col2.metric(
                        "Market prevalence",
                        f"{row['market_skill_prevalence_percent']:.2f}%",
                    )
                    col3.metric(
                        "Market skill rank",
                        (
                            int(row["market_skill_rank"])
                            if pd.notna(row["market_skill_rank"])
                            else "N/A"
                        ),
                    )

                    st.markdown("**Priority explanation**")
                    st.write(row["priority_explanation"])

                    st.markdown("**Evidence explanation**")
                    st.write(row["evidence_explanation"])

        st.download_button(
            "Download recommendations",
            data=recommendations.to_csv(index=False).encode("utf-8"),
            file_name="occupation_skill_recommendations.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# ML Evaluation
# ---------------------------------------------------------

elif page == "SHAP Explainability":
    st.header("SHAP Explainability")
    st.info(
        "These explanations describe the saved TF-IDF + Logistic Regression model's "
        "prediction that an occupation–skill pair was observed in the NYC snapshot. "
        "They are not causal explanations, worker skill-gap estimates, or forecasts. "
        "These are explanations on sampled data, not out-of-fold performance estimates."
    )

    shap_dir = DATA_DIR / "shap_v7"
    required = {
        "global": shap_dir / "global_feature_importance.csv",
        "predictions": shap_dir / "local_predictions.csv",
        "contributions": shap_dir / "local_feature_contributions.csv",
        "validation": shap_dir / "shap_validation.json",
    }
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required.values() if not p.exists()]
    if missing:
        st.error("Missing SHAP output file(s): " + ", ".join(missing))
        st.code("python src/explain_model_shap_v7.py", language="bash")
        st.caption("Generate the SHAP outputs in data/processed/shap_v7 and refresh this page.")
    else:
        import json
        import numpy as np

        global_df = pd.read_csv(required["global"])
        pred_df = pd.read_csv(required["predictions"])
        contrib_df = pd.read_csv(required["contributions"])
        with open(required["validation"], encoding="utf-8") as f:
            validation = json.load(f)

        needed_g = {"feature", "mean_abs_shap_log_odds"}
        needed_p = {"source_row_index", "occupation_label", "skill_label", "market_observed", "observed_probability", "base_log_odds", "predicted_log_odds", "sum_shap_log_odds"}
        needed_c = {"source_row_index", "feature", "shap_log_odds"}
        if (not needed_g.issubset(global_df.columns)
                or not needed_p.issubset(pred_df.columns)
                or not needed_c.issubset(contrib_df.columns)):
            st.error("SHAP output schema does not match v7. Regenerate the outputs using the v7 script.")
        elif pred_df.empty:
            st.warning("No local predictions in the SHAP outputs.")
        else:
            a, b, c = st.columns(3)
            a.metric("Explained samples", f"{len(pred_df):,}")
            b.metric("Model features", f"{int(validation.get('feature_count', 0)):,}")
            err = float(validation.get("max_additivity_error", float("nan")))
            c.metric("Max additivity error", f"{err:.2e}")
            if not np.isfinite(err) or err > 1e-5:
                st.warning("SHAP additivity check exceeds tolerance. Inspect the validation report.")

            st.subheader("Global feature importance")
            st.caption("Mean absolute SHAP contribution across the sampled examples, in log-odds. Magnitude shows influence, not direction or causality.")
            top_n = st.slider("Number of features", min_value=5, max_value=min(25, len(global_df)), value=min(15, len(global_df)), key="shap_top_n") if len(global_df)>=5 else len(global_df)
            global_view = global_df.sort_values("mean_abs_shap_log_odds", ascending=False).head(top_n)
            st.bar_chart(global_view.set_index("feature")["mean_abs_shap_log_odds"], horizontal=True)
            with st.expander("Global SHAP data"):
                st.dataframe(global_view, use_container_width=True, hide_index=True)
            png = shap_dir / "global_shap_importance.png"
            if png.exists():
                with st.expander("Original SHAP plot"):
                    st.image(str(png), use_container_width=True)

            st.subheader("Explain one occupation–skill prediction")
            occupations = sorted(pred_df["occupation_label"].fillna("Unknown").astype(str).unique())
            chosen_occ = st.selectbox("Occupation", occupations, key="shap_occ")
            subset = pred_df.loc[pred_df["occupation_label"].fillna("Unknown").astype(str) == chosen_occ].copy()
            subset["_label"] = subset["skill_label"].astype(str) + " · row " + subset["source_row_index"].astype(str)
            selected_label = st.selectbox("Skill / sampled record", subset["_label"].tolist(), key="shap_pair")
            record = subset.loc[subset["_label"] == selected_label].iloc[0]
            probability = float(record["observed_probability"])
            actual = int(record["market_observed"])
            # LogisticRegression.predict() uses the 0.5 probability decision
            # boundary for binary classification unless an external threshold
            # was selected. No custom decision threshold is stored in the
            # SHAP v7 outputs, so keep the assumption explicit in the UI.
            decision_threshold = 0.50
            predicted_observed = probability >= decision_threshold
            x, y, z = st.columns(3)
            x.metric("Predicted probability of observation", f"{probability:.1%}")
            y.metric("Predicted class (50% threshold)", "Observed" if predicted_observed else "Not observed")
            z.metric("Actually observed in NYC data", "Yes" if actual == 1 else "No")
            st.caption(
                "Classification uses a 50% probability threshold (default binary Logistic Regression rule). "
                "No custom threshold is specified in the saved SHAP metadata. "
                "An actual label of 0 means not observed in this processed NYC snapshot, "
                "not that a skill is unnecessary or unavailable. "
                "These are in-sample model explanations, not out-of-fold performance estimates."
            )

            identity = str(record["source_row_index"])
            local = contrib_df.loc[contrib_df["source_row_index"].astype(str) == identity].copy()
            if local.empty:
                st.warning("No local feature contributions were saved for this prediction.")
            else:
                local["shap_log_odds"] = pd.to_numeric(local["shap_log_odds"], errors="coerce")
                local = local.dropna(subset=["shap_log_odds"])
                positive = local[local["shap_log_odds"] > 0].nlargest(10, "shap_log_odds")
                negative = local[local["shap_log_odds"] < 0].nsmallest(10, "shap_log_odds")
                left, right = st.columns(2)
                with left:
                    st.markdown("**Features pushing toward observed**")
                    st.dataframe(positive[["feature", "shap_log_odds"]], hide_index=True, use_container_width=True)
                with right:
                    st.markdown("**Features pushing toward not observed**")
                    st.dataframe(negative[["feature", "shap_log_odds"]], hide_index=True, use_container_width=True)
                st.caption("Feature contributions are in log-odds, not percentage points. Negative values decrease and positive values increase the model score relative to its base value.")

            base = float(record["base_log_odds"])
            score = float(record["predicted_log_odds"])
            reconstructed = float(record["sum_shap_log_odds"])
            with st.expander("Prediction audit and source metadata"):
                st.write({"source_row_index": identity,
                          "occupation_uri": record.get("occupation_uri", ""),
                          "skill_uri": record.get("skill_uri", ""),
                          "base_log_odds": base,
                          "predicted_log_odds": score,
                          "base_plus_shap_log_odds": reconstructed,
                          "reconstruction_error": abs(score - reconstructed)})
                st.json(validation)
            st.download_button("Download local contributions (CSV)",
                               data=local.to_csv(index=False).encode("utf-8"),
                               file_name=f"shap_local_row_{identity}.csv", mime="text/csv")


elif page == "ML Evaluation":
    st.header("Machine-Learning Evaluation")

    st.markdown(
        """
        **Prediction objective:** estimate whether an ESCO occupation-skill
        relationship is observed in the available NYC job-posting dataset.

        This does not establish that a skill is universally required, predict
        an employment outcome, or forecast future demand.
        """
    )

    summary = require_data("ml_summary")
    baseline = require_data("baseline")
    features = require_data("features")

    if summary is not None:
        st.subheader("Occupation-aware grouped cross-validation")

        fold_results = summary[
            summary["result_type"] == "fold_mean"
        ].copy()

        if not fold_results.empty:
            fold_results = fold_results.rename(
                columns={
                    "metric": "Metric",
                    "value": "Mean",
                    "fold_std": "Fold standard deviation",
                }
            )

            st.dataframe(
                fold_results[["Metric", "Mean", "Fold standard deviation"]],
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Occupations are separated across folds. These estimates "
                "evaluate generalisation to held-out occupations under the "
                "sampled evaluation dataset."
            )

    if baseline is not None:
        st.subheader("Model versus majority-class baseline")

        pivot = baseline.pivot(
            index="metric",
            columns="model",
            values="mean",
        )

        preferred_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]

        available_metrics = [
            metric for metric in preferred_metrics if metric in pivot.index
        ]

        st.dataframe(
            pivot.loc[available_metrics].reset_index(),
            hide_index=True,
            width="stretch",
        )

    if features is not None:
        st.subheader("Global model feature explanations")

        direction_options = sorted(features["direction"].dropna().unique())

        selected_direction = st.selectbox(
            "Feature direction",
            direction_options,
        )

        feature_data = features[
            features["direction"] == selected_direction
        ].copy()

        feature_data = feature_data.sort_values(
            "absolute_coefficient",
            ascending=False,
        )

        st.dataframe(
            feature_data[
                [
                    "feature",
                    "coefficient",
                    "associated_class",
                    "interpretation",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "These are global linear-model feature associations, not causal "
            "effects or explanations for an individual prediction."
        )


# ---------------------------------------------------------
# ML-Assisted Decision Support
# ---------------------------------------------------------

elif page == "ML-Assisted Decision Support":
    st.header("ML-Assisted Career Decision Support")

    st.markdown(
        "Explore model estimates and existing career-gap recommendations "
        "as separate evidence views. A model score is not a rule-based priority."
    )

    st.warning(
        "The model estimates whether an ESCO occupation-skill relationship "
        "was observed in the available NYC job-posting dataset. It does not "
        "predict future demand or prove that a skill is required. Scores are "
        "exploratory and are not guaranteed to be calibrated probabilities."
    )

    mode = st.radio(
        "Choose an analysis",
        [
            "ESCO model exploration",
            "Career-gap recommendations",
        ],
        horizontal=True,
        key="ml_decision_mode",
    )

    demand_data = require_data("demand")
    career_data = require_data("career")

    def normalized(series):
        return series.astype(str).str.strip().str.casefold()

    if mode == "ESCO model exploration":
        model_data = require_data("model_dataset")
        model_path = DATA_DIR / "market_observation_model_grouped.joblib"

        if model_data is not None:
            if not model_path.exists():
                st.error(
                    "The saved grouped model is missing: "
                    "data/processed/market_observation_model_grouped.joblib"
                )
            else:
                try:
                    model = joblib.load(model_path)
                except Exception as exc:
                    model = None
                    st.error(f"Could not load the saved model: {exc}")

                required = {
                    "occupation_uri", "occupation_label",
                    "skill_uri", "skill_label",
                    "relation_type", "skill_type",
                }

                if not required.issubset(model_data.columns):
                    st.error(
                        "Missing model input columns: "
                        + ", ".join(sorted(required - set(model_data.columns)))
                    )
                elif model is not None:
                    candidates = model_data.dropna(
                        subset=[
                            "occupation_label", "skill_label",
                            "relation_type", "skill_type",
                        ]
                    ).copy()

                    occupations = sorted(
                        candidates["occupation_label"].unique(),
                        key=str.casefold,
                    )

                    selected_occupation = st.selectbox(
                        "Choose an ESCO occupation",
                        occupations,
                        key="ml_model_occupation",
                    )

                    occupation_rows = candidates[
                        candidates["occupation_label"] == selected_occupation
                    ]

                    skill_options = sorted(
                        occupation_rows["skill_label"].unique(),
                        key=str.casefold,
                    )

                    selected_skill = st.selectbox(
                        "Choose an ESCO skill",
                        skill_options,
                        key="ml_model_skill",
                    )

                    pair_rows = occupation_rows[
                        occupation_rows["skill_label"] == selected_skill
                    ]

                    row = pair_rows.iloc[0]
                    model_input = (
                        f"{row['occupation_label']} "
                        f"{row['skill_label']} "
                        f"{row['skill_type']} "
                        f"{row['relation_type']}"
                    )

                    try:
                        probabilities = model.predict_proba([model_input])[0]
                        classes = list(model.classes_)
                        positive_index = classes.index(1)
                        score = float(probabilities[positive_index])
                    except Exception as exc:
                        score = None
                        st.error(f"Could not score this pair: {exc}")

                    st.subheader("Evidence for the selected ESCO pair")

                    if score is not None:
                        st.metric("Model-estimated score", f"{score:.1%}")
                    else:
                        st.metric("Model-estimated score", "Unavailable")

                    # URI-first matching avoids confusing identical labels
                    # that refer to different ESCO concepts.
                    demand_rows = pd.DataFrame()
                    if demand_data is not None:
                        if (
                            "occupation_uri" in demand_data.columns
                            and "skill_uri" in demand_data.columns
                        ):
                            demand_rows = demand_data[
                                demand_data["occupation_uri"].eq(row["occupation_uri"])
                                & demand_data["skill_uri"].eq(row["skill_uri"])
                            ]

                    if not demand_rows.empty:
                        drow = demand_rows.iloc[0]
                        a, b = st.columns(2)
                        count = drow.get("skill_posting_count")
                        prevalence = drow.get("skill_prevalence_percent")

                        a.metric(
                            "Observed posting count",
                            f"{int(count):,}" if pd.notna(count) else "Unavailable",
                        )
                        b.metric(
                            "Observed posting prevalence",
                            f"{prevalence:.1f}%"
                            if pd.notna(prevalence) else "Unavailable",
                        )
                    else:
                        st.info(
                            "No matching occupation–skill URI was found in the "
                            "loaded market-demand table. This is missing evidence, "
                            "not proof of zero demand."
                        )

                    st.markdown("**Model input used**")
                    st.code(model_input, language=None)

                    st.caption(
                        "This is saved-model inference, not a held-out evaluation "
                        "result. See ML Evaluation for occupation-aware "
                        "cross-validation metrics. The displayed score is not "
                        "a calibrated probability unless calibration has been "
                          "independently established."
                    )

    else:
        if career_data is not None and not career_data.empty:
            required_career = {
                "occupation_label", "skill_label", "priority",
                "market_skill_posting_count",
                "market_skill_prevalence_percent",
                "evidence_explanation",
            }

            if not required_career.issubset(career_data.columns):
                st.error(
                    "The career-decision table is missing columns: "
                    + ", ".join(sorted(required_career - set(career_data.columns)))
                )
            else:
                occupations = sorted(
                    career_data["occupation_label"].dropna().unique(),
                    key=str.casefold,
                )

                selected_occupation = st.selectbox(
                    "Choose a target occupation",
                    occupations,
                    key="ml_career_occupation",
                )

                occupation_gaps = career_data[
                    career_data["occupation_label"] == selected_occupation
                ].copy()

                selected_skill = st.selectbox(
                    "Choose a prioritised skill gap",
                    sorted(occupation_gaps["skill_label"].dropna().unique(),
                           key=str.casefold),
                    key="ml_career_skill",
                )

                selected_rows = occupation_gaps[
                    occupation_gaps["skill_label"] == selected_skill
                ]

                if selected_rows.empty:
                    st.warning("No recorded priority row exists for this selection.")
                else:
                    gap = selected_rows.iloc[0]
                    st.subheader("Recorded career-gap evidence")

                    a, b, c = st.columns(3)
                    count = gap["market_skill_posting_count"]
                    prevalence = gap["market_skill_prevalence_percent"]

                    a.metric(
                        "Observed posting count",
                        f"{int(count):,}" if pd.notna(count) else "Unavailable",
                    )
                    b.metric(
                        "Observed posting prevalence",
                        f"{prevalence:.1f}%"
                        if pd.notna(prevalence) else "Unavailable",
                    )
                    c.metric("Existing rule-based priority", str(gap["priority"]))

                    st.markdown("**Evidence explanation**")
                    st.write(gap["evidence_explanation"])

                    st.info(
                        "No model score is shown for this skill gap because its "
                        "label does not match a pair in the current ESCO model "
                        "dataset. The priority above is the existing rule-based "
                        "classification, not a machine-learning prediction."
                    )

    with st.expander("How to interpret the evidence"):
        st.markdown(
            """
            - **Model estimate:** output of the saved grouped model for an
              ESCO occupation–skill input.
            - **Observed demand:** evidence from the available NYC job-posting
              snapshot, where an occupation–skill URI match exists.
            - **Rule-based priority:** the previously computed classification
              for one of the existing target-occupation skill gaps.
            - **Missing evidence:** means no matching record was found in the
              loaded table. It does not establish that a skill is absent from
              the wider labour market.
            - **Scope:** the model estimates market observation; it does not
              establish a universal skill requirement or predict future demand.
            """
        )


# Personalised Skill Gaps
# ---------------------------------------------------------

elif page == "Personalised Skill Gaps":
    st.header("Personalised Skill Gap Analysis")

    personal = require_data("personal")

    if personal is not None:
        users = sorted(personal["user_id"].dropna().unique())

        selected_user = st.selectbox(
            "Select a user profile",
            users,
        )

        user_data = personal[
            personal["user_id"] == selected_user
        ].copy()

        occupations = sorted(
            user_data["occupation_label"].dropna().unique()
        )

        if occupations:
            selected_occupation = st.selectbox(
                "Select target occupation",
                occupations,
            )

            user_data = user_data[
                user_data["occupation_label"] == selected_occupation
            ]

        gaps_only = st.checkbox(
            "Show only identified personal gaps",
            value=True,
        )

        if gaps_only:
            user_data = user_data[user_data["personal_gap"] == True]

        col1, col2, col3 = st.columns(3)

        col1.metric("Displayed skills", len(user_data))
        col2.metric(
            "Identified gaps",
            int((user_data["personal_gap"] == True).sum()),
        )
        col3.metric(
            "Market-observed gaps",
            int(
                (
                    (user_data["personal_gap"] == True)
                    & user_data["skill_posting_count"].notna()
                ).sum()
            ),
        )

        st.dataframe(
            user_data[
                [
                    "occupation_label",
                    "skill_label",
                    "user_skill_status",
                    "personal_gap",
                    "evidence_basis",
                    "skill_posting_count",
                    "skill_prevalence_percent",
                    "demand_category",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "The displayed profile is the existing user profile in the dataset; "
            "this page does not yet collect or save new user-entered skills."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Research prototype | Results depend on the available ESCO taxonomy, "
    "NYC job-posting snapshot, data-processing rules, and evaluation sampling."
)