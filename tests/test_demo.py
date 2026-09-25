from sstbr.demo.synthetic import run_demo


def test_full_synthetic_demo() -> None:
    result = run_demo()
    assert result["representation_shape"] == (6, 331)
    assert result["context_shape"] == (1987,)
    assert result["evaluation_samples"] == 3
