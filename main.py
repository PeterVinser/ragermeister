import argparse
from dotenv import load_dotenv

from solution.eval.comparison import main as run_comparison


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="store_true",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
    )
    args = parser.parse_args()
    run_comparison(use_llm_extractor=args.llm, resolve=args.resolve)


if __name__ == "__main__":
    main()
