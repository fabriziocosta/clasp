def main(argv: list[str] | None = None) -> int:
    from clasp.cli.main import main as _main

    return _main(argv)

__all__ = ["main"]
