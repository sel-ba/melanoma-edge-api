#!/usr/bin/env python
"""Run the full HAM10000 validation suite using Great Expectations.

Usage:
    python -m src.validation.run_suite          # default paths
    python -m src.validation.run_suite --metadata data/raw/HAM10000_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.validation.expectations.create_suite import (
    build_expectation_suite,
    run_checkpoint,
)
from src.validation.image_integrity import validate_image_directory
from src.validation.duplicate_detector import find_visual_duplicates

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full HAM10000 validation suite.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/HAM10000_metadata.csv",
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/raw",
        help="Path to raw image directory.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory.",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip perceptual hash duplicate check (slow).",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[2]
    metadata_path = project_root / args.metadata
    image_dir = project_root / args.image_dir

    report: dict[str, object] = {}

    # 1. Great Expectations metadata validation
    print("\n" + "=" * 60)
    print("STEP 1: Great Expectations Metadata Validation")
    print("=" * 60)
    try:
        if not metadata_path.exists():
            print(f"WARNING: Metadata not found at {metadata_path}. Skipping GE validation.")
            report["ge_validation"] = {"status": "skipped", "reason": "metadata_missing"}
        else:
            build_expectation_suite(str(metadata_path), project_root=project_root)
            ge_result = run_checkpoint(project_root=project_root)
            report["ge_validation"] = {"status": "completed", "success": ge_result.get("success")}
    except Exception as exc:
        print(f"GE validation error: {exc}")
        report["ge_validation"] = {"status": "error", "error": str(exc)}

    # 2. Image integrity check
    print("\n" + "=" * 60)
    print("STEP 2: Image Integrity Check")
    print("=" * 60)
    try:
        integrity = validate_image_directory(str(image_dir))
        report["image_integrity"] = {
            "total": integrity["total"],
            "valid": integrity["valid"],
            "corrupt": integrity["corrupt"],
        }
        print(f"  Valid: {integrity['valid']}/{integrity['total']}")
    except Exception as exc:
        print(f"Image integrity error: {exc}")
        report["image_integrity"] = {"status": "error", "error": str(exc)}

    # 3. Duplicate detection (optional, slow)
    if not args.skip_duplicates:
        print("\n" + "=" * 60)
        print("STEP 3: Duplicate Detection")
        print("=" * 60)
        try:
            duplicates = find_visual_duplicates(str(image_dir))
            report["duplicates"] = {
                "groups": len(duplicates),
                "status": "completed",
            }
            print(f"  Duplicate groups found: {len(duplicates)}")
        except Exception as exc:
            print(f"Duplicate detection error: {exc}")
            report["duplicates"] = {"status": "error", "error": str(exc)}

    # Save summary report
    report_path = project_root / "reports" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nValidation report saved to {report_path}")


if __name__ == "__main__":
    main()
