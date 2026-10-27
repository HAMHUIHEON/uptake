from forecaster.batch import model_for, build_arg_parser


def test_model_flags():
    p = build_arg_parser()
    a = p.parse_args(["--manifest", "m", "--out", "o", "--private-model", "muse-spark-1.3-contributor"])
    assert a.private_model == "muse-spark-1.3-contributor" and a.public_model == "muse-spark-1.3-contributor"
    assert model_for({"public": False}, private_model=a.private_model) == "muse-spark-1.3-contributor"
    d = p.parse_args(["--manifest", "m", "--out", "o"])
    assert d.private_model == "muse-spark-1.3"
