"""
Command-line interface for PACMOF2.

Usage examples:
    # Single neutral MOF
    pacmof2 path/to/file.cif -o output_dir/

    # Multiple neutral MOFs
    pacmof2 path/to/cifs/ -o output_dir/ --multiple

    # Single ionic MOF
    pacmof2 path/to/file.cif -o output_dir/ --net-charge -2

    # Multiple ionic MOFs with net charges from JSON
    pacmof2 path/to/cifs/ -o output_dir/ --multiple --net-charges net_charges.json
"""

import argparse
import json
import sys
from typing import Union

from pacmof2 import __version__


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pacmof2",
        description=(
            "PACMOF2: Predict partial atomic charges in Metal-Organic Frameworks "
            "with DFT-level accuracy using pre-trained ML models."
        ),
        epilog=(
            "Examples:\n"
            "  pacmof2 my_mof.cif -o output/\n"
            "  pacmof2 cifs/ -o output/ --multiple\n"
            "  pacmof2 my_mof.cif -o output/ --net-charge -2\n"
            "  pacmof2 cifs/ -o output/ --multiple --net-charges charges.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to a CIF file or a directory of CIF files (with --multiple).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for CIF files with predicted charges.",
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        default=False,
        help="Process all CIF files in the input directory.",
    )
    parser.add_argument(
        "--identifier",
        default="_pacmof",
        help="Suffix appended to output filenames (default: '_pacmof').",
    )
    parser.add_argument(
        "--adjust-method",
        choices=["mean", "magnitude"],
        default="mean",
        help="Charge adjustment method to enforce net charge (default: 'mean').",
    )

    # Ionic MOF options (mutually exclusive for single vs batch)
    charge_group = parser.add_mutually_exclusive_group()
    charge_group.add_argument(
        "--net-charge",
        type=float,
        default=None,
        help="Net charge for a single ionic MOF (e.g., -2). Omit for neutral MOFs.",
    )
    charge_group.add_argument(
        "--net-charges",
        type=str,
        default=None,
        help=(
            "Path to a JSON file mapping CIF filenames to net charges "
            "(for batch ionic MOF processing)."
        ),
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the PACMOF2 CLI."""
    args = parse_args(argv)

    # Resolve net charge argument
    net_charge: Union[int, float, dict] = 0
    if args.net_charge is not None:
        net_charge = args.net_charge
    elif args.net_charges is not None:
        try:
            with open(args.net_charges, "r") as f:
                net_charge = json.load(f)
        except FileNotFoundError:
            print(f"Error: Net charges file not found: {args.net_charges}")
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {args.net_charges}: {e}")
            return 1

        if not isinstance(net_charge, dict):
            print(
                f"Error: {args.net_charges} must contain a JSON object "
                f"mapping filenames to net charges."
            )
            return 1

    # Import here to avoid slow import on --help / --version
    from pacmof2.pacmof2 import get_charges

    try:
        get_charges(
            path_to_cif=args.input,
            output_path=args.output,
            identifier=args.identifier,
            multiple_cifs=args.multiple,
            adjust_charge_method=args.adjust_method,
            net_charge=net_charge,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
