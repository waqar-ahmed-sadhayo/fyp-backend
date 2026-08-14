from app.ml import explain, predictor


def test_explain_returns_ranked_contributions_or_none():
    """Every disease should either get a fast, ranked explanation (LR/RF/GB/
    HistGB-chosen models) or explicitly None (SVM-chosen — no fast exact
    SHAP algorithm, see explain.py's docstring for why that's a deliberate
    tradeoff rather than an oversight)."""
    for disease in predictor.DISEASES:
        result = explain.explain(disease, {})
        chosen_model = predictor.get_metrics(disease)["chosen_model"]

        if chosen_model == "svm":
            assert result is None, disease
            continue

        assert result is not None, disease
        assert 1 <= len(result) <= explain.TOP_N
        contributions = [abs(item["contribution"]) for item in result]
        assert contributions == sorted(contributions, reverse=True)
        for item in result:
            assert set(item.keys()) == {"feature", "value", "contribution"}


def test_explain_is_fast():
    """Regression guard for the ~17s/request PermutationExplainer mistake
    this module started with — every disease's explanation (or the SVM
    None-shortcut) should return well under a second."""
    import time

    for disease in predictor.DISEASES:
        start = time.monotonic()
        explain.explain(disease, {})
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"{disease} explanation took {elapsed:.2f}s"
