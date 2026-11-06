from training.ray.recover_tsl_checkpoints import is_final, parse_setting

BASE = "long_term_forecast_{ds}_192_{h}_s{s}_{m}_custom_ftM_sl192_ll96_pl{h}_dm32_nh8_el3_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_{label}"


def test_parse_and_classify():
    d = BASE.format(ds="spec4", h=96, s=44, m="PatchTST", label="tuned-PatchTST_0")
    assert parse_setting(d) == {"dataset": "spec4", "horizon": 96, "seed": 44, "model": "PatchTST", "label": "tuned-PatchTST"}
    assert is_final(d)
    assert is_final(BASE.format(ds="spec151", h=12, s=2021, m="Autoformer", label="electrosense_0"))
    assert not is_final(BASE.format(ds="spec25", h=192, s=42, m="Informer", label="tune-Informer-c3_0"))
    assert not is_final("not_a_setting")
