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


def test_alphas_are_the_resta_sweep_with_a_fine_low_end_and_a_break_control():
    """SafeTune's own docs report that on Tiny Aya 0.1 and 0.25 restore refusal and 0.5 or more breaks the
    model, and our A1 already holds the Base-to-Global step once. The grid is dense where ReSta can work and
    keeps 1.0 as a deliberate break control (docs/toolkit-notes.md, "The alpha grid")."""
    assert config.ALPHAS == (0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0)
    assert list(config.ALPHAS) == sorted(set(config.ALPHAS))  # increasing, no repeats
    assert config.ALPHAS[0] == 0.0 and config.ALPHAS[-1] == 1.0
    assert 0.75 not in config.ALPHAS  # the blend uses t in (0.25, 0.5, 0.75); the ReSta grid is not that


def test_every_alpha_survives_the_path_name_round_trip():
    from msd import gen

    names = [gen.format_alpha(a) for a in config.ALPHAS]
    assert len(set(names)) == len(names)  # no two alphas share a folder
    for alpha, name in zip(config.ALPHAS, names):
        assert gen.canonical_alpha(alpha) == alpha  # not rounded away
        assert float(name) == alpha  # the folder name reads back as the same number
        assert "e" not in name.lower() and not name.endswith(".")  # never exponent notation
    assert names == ["0.0", "0.05", "0.1", "0.15", "0.25", "0.35", "0.5", "1.0"]
