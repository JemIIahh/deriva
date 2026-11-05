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


def test_dev_benign_n_is_the_dev_split_benign_size():
    # dev split = PROBE_N harmful + DEV_BENIGN_N benign-edgy prompts per language
    assert config.DEV_BENIGN_N == 100
