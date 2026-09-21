"""Verify imports and run the synthetic demonstration."""

from run_demo import run_demo


def main() -> None:
    result = run_demo()
    if result["evaluation_samples"] != 3:
        raise RuntimeError("Unexpected synthetic evaluation cardinality.")
    print("SST-BR partial installation: PASS")


if __name__ == "__main__":
    main()
