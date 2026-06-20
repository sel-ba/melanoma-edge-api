from __future__ import annotations

from pathlib import Path

import great_expectations as gx
import pandas as pd

EXPECTED_COLUMNS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
VALID_DIAGNOSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
SUITE_NAME = "ham10000_suite"
CHECKPOINT_NAME = "ham10000_checkpoint"


def _get_context(project_root: Path | None = None) -> gx.FileDataContext:
    """Return a Great Expectations FileDataContext using the fluent API."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    ge_dir = project_root / "great_expectations"
    ge_dir.mkdir(exist_ok=True)

    config_yml = ge_dir / "great_expectations.yml"
    if not config_yml.exists():
        raise FileNotFoundError(
            f"GX config not found at {config_yml}. Create great_expectations.yml first."
        )

    return gx.get_context(context_root_dir=str(ge_dir))


def _get_or_create_batch_request(
    context: gx.FileDataContext, metadata_path: Path
) -> dict:
    """Create a pandas datasource + asset and return batch request.

    Uses the GE 0.18 fluent API: context.sources.add_pandas().
    The asset is recreated each time because the dataframe is not persisted.
    """
    df = pd.read_csv(metadata_path)

    datasource_name = "ham10000_pandas"
    asset_name = "metadata"

    try:
        datasource = context.get_datasource(datasource_name)
    except Exception:
        datasource = context.sources.add_pandas(datasource_name)

    # Always delete and recreate the asset with fresh dataframe
    # because GE fluent API does not persist the in-memory dataframe
    try:
        datasource.delete_asset(asset_name)
    except Exception:
        pass

    asset = datasource.add_dataframe_asset(name=asset_name, dataframe=df)
    return asset.build_batch_request()


def build_expectation_suite(
    metadata_path: str | Path,
    project_root: Path | None = None,
    suite_name: str = SUITE_NAME,
) -> gx.ExpectationSuite:
    """Create or update the HAM10000 expectation suite (GE 0.18 fluent API)."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    context = _get_context(project_root)
    metadata_path = Path(metadata_path)
    batch_request = _get_or_create_batch_request(context, metadata_path)

    # Remove old suite if present, then create fresh
    try:
        context.delete_expectation_suite(suite_name)
    except Exception:
        pass

    context.add_expectation_suite(suite_name)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # --- Table-level ---
    validator.expect_table_columns_to_match_ordered_list(EXPECTED_COLUMNS)
    validator.expect_table_row_count_to_be_between(min_value=10000, max_value=10100)

    # --- Column-level ---
    validator.expect_column_values_to_not_be_null("image_id")
    validator.expect_column_values_to_not_be_null("dx")
    validator.expect_column_values_to_be_in_set("dx", VALID_DIAGNOSES)
    validator.expect_column_values_to_be_between(
        "age", min_value=0, max_value=120, mostly=0.95
    )
    validator.expect_column_values_to_be_unique("image_id")

    validator.save_expectation_suite(discard_failed_expectations=False)
    n = len(validator.get_expectation_suite().expectations)
    print(f"  Suite '{suite_name}' saved with {n} expectations")
    return validator.get_expectation_suite()


def run_checkpoint(
    checkpoint_name: str = CHECKPOINT_NAME,
    project_root: Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict:
    """Run the GE checkpoint and return results as dict."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    context = _get_context(project_root)

    if metadata_path is None:
        metadata_path = project_root / "data" / "raw" / "HAM10000_metadata.csv"

    batch_request = _get_or_create_batch_request(context, Path(metadata_path))

    # Ensure expectation suite exists
    try:
        context.get_expectation_suite(SUITE_NAME)
    except Exception:
        build_expectation_suite(str(metadata_path), project_root=project_root)

    # Create or update checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": SUITE_NAME,
        }],
        run_name_template="ham10000_validation_%Y%m%d-%H%M%S",
    )

    result = checkpoint.run()

    run_id = result.list_validation_results()[0].meta["run_id"]
    success = result.success

    print(f"  Checkpoint '{checkpoint_name}' completed.")
    print(f"  Run ID: {run_id.run_name}")
    print(f"  Success: {success}")
    print("  Data Docs: great_expectations/uncommitted/data_docs/local_site/index.html")

    # Print per-expectation results
    validation_results = result.list_validation_results()[0]["results"]
    for vr in validation_results:
        cfg = vr.expectation_config
        name = cfg.kwargs.get("column") or cfg.expectation_type
        status = "PASS" if vr.success else "FAIL"
        print(f"    [{status}] {cfg.expectation_type}: {name}")

    return {
        "success": success,
        "run_id": run_id.run_name,
        "statistics": result.list_validation_results()[0]["statistics"],
    }


def validate_metadata(
    metadata_path: str | Path,
    project_root: Path | None = None,
) -> dict:
    """Full pipeline: build suite and run checkpoint."""
    build_expectation_suite(metadata_path, project_root=project_root)
    return run_checkpoint(project_root=project_root, metadata_path=metadata_path)
