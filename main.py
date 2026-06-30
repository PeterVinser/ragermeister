import argparse
from dotenv import load_dotenv

from solution.eval.comparison import main as run_comparison


def main() -> None:
    load_dotenv()

    run_comparison()


if __name__ == "__main__":
    main()
