from pathlib import Path

import pandas as pd

from prepare_energy_data import build_parser, chronological_split, prepare_dataset


def test_parser_accepts_custom_paths() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "source.txt",
            "--processed-dir",
            "generated",
            "--reports-dir",
            "summaries",
        ]
    )

    assert args.input == Path("source.txt")
    assert args.processed_dir == Path("generated")
    assert args.reports_dir == Path("summaries")


def test_chronological_split_keeps_order() -> None:
    frame = pd.DataFrame({"value": range(20)})

    train, validation, test = chronological_split(frame)

    assert train["value"].tolist() == list(range(14))
    assert validation["value"].tolist() == [14, 15, 16]
    assert test["value"].tolist() == [17, 18, 19]


def test_missing_custom_input_reports_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    try:
        prepare_dataset(missing, tmp_path / "processed", tmp_path / "reports")
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("prepare_dataset accepted a missing input file")
