import argparse

from ms_pred import __version__
from ms_pred.model_registry import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(prog="python -m ms_pred")
    parser.add_argument("--version", action="store_true", help="show package and model versions")
    args = parser.parse_args()
    if not args.version:
        parser.print_help()
        return
    print(f"ms_pred {__version__}")
    for name, info in MODEL_REGISTRY.items():
        print(f"{name} {info['version']}")


if __name__ == "__main__":
    main()
