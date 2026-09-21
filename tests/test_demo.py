from scripts.run_demo import run_demo


def test_partial_synthetic_demo() -> None:
    result = run_demo()
    assert result["local_windows"] == 6
    assert result["evaluation_samples"] == 3
    assert result["metrics"]["RMSE"] > 10
