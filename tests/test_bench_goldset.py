from bench.goldset import tokenize, entity_in_text, score_recall


def test_tokenize_splits_on_punct_and_casefolds():
    assert tokenize("PI3K-Akt pathway") == ["pi3k", "akt", "pathway"]
    assert tokenize("TGF-β") == ["tgf", "β"]


def test_entity_in_text_contiguous_subsequence():
    text = "We propose the PI3K-Akt signaling pathway drives growth."
    assert entity_in_text("PI3K-Akt", text)
    assert entity_in_text("PI3K Akt signaling", text)
    assert not entity_in_text("Akt PI3K", text)        # order matters
    assert not entity_in_text("mTOR", text)


def test_entity_in_text_handles_greek_and_hyphen_variants():
    assert entity_in_text("TGF-β", "role of TGF β in fibrosis")
    assert entity_in_text("dimethyl-fumarate", "treated with dimethyl fumarate daily")


def _h(text):
    from types import SimpleNamespace
    return SimpleNamespace(text=text, summary="", elo_rating=1200.0)


def test_score_recall_pool_and_topk():
    hyps = [
        _h("PI3K-Akt drives it"),
        _h("also TGF-β matters"),
        _h("irrelevant"),
    ]
    gold = ["PI3K-Akt", "TGF-β", "mTOR"]
    assert score_recall(hyps, gold) == 2 / 3          # pool recall
    assert score_recall(hyps[:1], gold) == 1 / 3


def test_score_recall_empty_gold_is_zero():
    assert score_recall([_h("x")], []) == 0.0


def test_score_recall_k_selects_top_elo():
    from types import SimpleNamespace
    def h(text, elo):
        return SimpleNamespace(text=text, summary="", elo_rating=elo)
    hyps = [
        h("PI3K-Akt drives it", elo=1000.0),   # low elo
        h("TGF-β matters", elo=1500.0),         # high elo
    ]
    gold = ["PI3K-Akt", "TGF-β"]
    # k=1 searches only the top-elo hypothesis (TGF-β) → 1/2 recall
    assert score_recall(hyps, gold, k=1) == 0.5
    # k=2 searches both → 2/2
    assert score_recall(hyps, gold, k=2) == 1.0
