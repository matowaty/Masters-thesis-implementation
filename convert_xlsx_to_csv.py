"""
convert_xlsx_to_csv.py — One-time XLSX → CSV converter

Reads every .xlsx file in DATA/, locates the 'Exchange Date' header row
used by Refinitiv/LSEG exports, extracts the OHLCV time-series data,
and writes a clean CSV file alongside the original.

The resulting CSVs have:
    - A 'Datetime' column (ISO-8601 formatted) as the first column.
    - Columns: Datetime, Open, High, Low, Close, Volume
    - Rows sorted in ascending chronological order.
    - No metadata / VAP / statistics rows.

Usage:
    python convert_xlsx_to_csv.py            # converts all files in DATA/
    python convert_xlsx_to_csv.py --data-dir path/to/dir   # custom directory
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def find_header_row(filepath: Path, max_scan: int = 60) -> int:
    """Scan an XLSX to locate the row containing 'Exchange Date'."""
    probe = pd.read_excel(filepath, header=None, nrows=max_scan)
    for i in range(len(probe)):
        cell = probe.iloc[i, 0]
        if pd.notna(cell) and "Exchange Date" in str(cell):
            return i
    raise ValueError(
        f"Could not locate 'Exchange Date' header within first "
        f"{max_scan} rows of {filepath}"
    )


def convert_single_file(xlsx_path: Path, output_dir: Path | None = None) -> Path:
    """Convert one Refinitiv .xlsx export to a clean CSV.

    Args:
        xlsx_path: Path to the source .xlsx file.
        output_dir: Directory for the output CSV. Defaults to the same
                    directory as the source file.

    Returns:
        Path to the newly created .csv file.
    """
    if output_dir is None:
        output_dir = xlsx_path.parent

    header_row = find_header_row(xlsx_path)
    logger.debug("Header at row %d in %s", header_row, xlsx_path.name)

    df = pd.read_excel(xlsx_path, header=header_row)

    # Drop any trailing unnamed columns from the export
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Build a proper datetime from 'Exchange Date'
    df["Datetime"] = pd.to_datetime(df["Exchange Date"])
    df = df.set_index("Datetime")

    # Keep only OHLCV columns
    df = df[_OHLCV_COLUMNS].copy()

    # Refinitiv exports are in reverse chronological order — flip
    df = df.sort_index(ascending=True)

    # Write CSV with Datetime as a regular column
    csv_path = output_dir / f"{xlsx_path.stem}.csv"
    df.to_csv(csv_path, index=True)

    logger.info(
        "[OK] %s -> %s  (%d rows, %s -> %s)",
        xlsx_path.name,
        csv_path.name,
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return csv_path


def convert_directory(data_dir: Path) -> list[Path]:
    """Convert every .xlsx file in *data_dir* to CSV.

    Args:
        data_dir: Directory containing .xlsx files.

    Returns:
        List of paths to the created CSV files.
    """
    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.warning("No .xlsx files found in %s", data_dir)
        return []

    logger.info("Found %d .xlsx files in %s", len(xlsx_files), data_dir)
    csv_paths: list[Path] = []

    for xlsx_path in xlsx_files:
        try:
            csv_path = convert_single_file(xlsx_path)
            csv_paths.append(csv_path)
        except Exception as exc:
            logger.error("[FAIL] Failed to convert %s: %s", xlsx_path.name, exc)

    logger.info(
        "Conversion complete -- %d / %d files converted successfully",
        len(csv_paths),
        len(xlsx_files),
    )
    return csv_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Refinitiv/LSEG .xlsx exports to clean CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "DATA",
        help="Directory containing .xlsx files (default: DATA/)",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error("Directory does not exist: %s", args.data_dir)
        sys.exit(1)

    convert_directory(args.data_dir)


if __name__ == "__main__":
    main()
