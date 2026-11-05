from msd import config


def test_spec_constants():
    assert config.LANGS == ("en", "hi", "hinglish", "ar", "sw")
    assert "en" not in config.NON_ENGLISH
    assert config.GATE_MIN_RISE == 0.10
    assert config.GATE_MIN_LANGS == 2
    assert config.PROBE_N == 150
    assert config.REPORT_HARMFUL_N == 300
    assert config.REPORT_BENIGN_N == 200
    assert config.RELIABILITY_N == 60
    assert config.KAPPA_LOW == 0.6


def test_model_ids_are_owner_slash_name_strings():
    import re

    assert {"A0", "BASE"} <= set(config.MODEL_IDS)
    for arm in ("A0", "BASE"):
        repo_id = config.MODEL_IDS[arm]
        assert isinstance(repo_id, str) and repo_id
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id), repo_id
    assert config.MODEL_IDS["A0"] != config.MODEL_IDS["BASE"]
    # Verified against the public Hugging Face API on 2026-11-05 (see the comment in config.py).
    assert config.MODEL_IDS["A0"] == "CohereLabs/tiny-aya-global"
    assert config.MODEL_IDS["BASE"] == "CohereLabs/tiny-aya-base"
    assert set(config.MODEL_IDS) <= set(config.ARMS)


def test_dev_benign_n_is_the_dev_split_benign_size():
    # dev split = PROBE_N harmful + DEV_BENIGN_N benign-edgy prompts per language
    assert config.DEV_BENIGN_N == 100
