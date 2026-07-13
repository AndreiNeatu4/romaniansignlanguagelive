"""
Complete Training Pipeline
==========================

This script runs the entire training pipeline in sequence:
1. Extract augmented data from videos
2. Extract static images (JPG/PNG) - skipped for moving letters (J/Z/Q/S/Ș/Ț)
3. Prepare dataset (combine .npy files into dataset.pkl)
4. Train the model

Usage:
    python run_pipeline.py           # Run all steps
    python run_pipeline.py --skip-extract   # Skip video extraction
    python run_pipeline.py --skip-images    # Skip static image extraction
    python run_pipeline.py --skip-prepare   # Skip dataset preparation
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_step(step_name: str, script_path: str):
    """
    Run a pipeline step.

    Returns:
        (success: bool, elapsed: timedelta)
    """
    print("\n" + "=" * 70)
    print(f"  STEP: {step_name}")
    print("=" * 70 + "\n")

    start_time = datetime.now()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(__file__).parent)
    )

    elapsed = datetime.now() - start_time

    if result.returncode == 0:
        print(f"\n[OK] {step_name} completed in {elapsed}")
        return True, elapsed
    else:
        print(f"\n[ERROR] {step_name} failed with code {result.returncode}")
        return False, elapsed


def format_duration(td) -> str:
    total_seconds = int(td.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_timing_summary(stage_timings):
    print("\n" + "=" * 70)
    print("  STAGE TIMING SUMMARY")
    print("=" * 70)

    name_width = max((len(name) for name, _, _ in stage_timings), default=10)
    for name, status, elapsed in stage_timings:
        if status == "skipped":
            print(f"  [SKIPPED] {name:<{name_width}}  --")
        else:
            tag = "[OK]   " if status == "ok" else "[FAIL] "
            print(f"  {tag} {name:<{name_width}}  {format_duration(elapsed)}")


def main():
    parser = argparse.ArgumentParser(description='Run complete training pipeline')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip video extraction step')
    parser.add_argument('--skip-images', action='store_true',
                        help='Skip static image extraction step')
    parser.add_argument('--skip-prepare', action='store_true',
                        help='Skip dataset preparation step')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training step')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  GESTURE RECOGNITION - COMPLETE TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    base_dir = Path(__file__).parent

    steps = [
        ("1. Extract Augmented Video Data",
         base_dir / "data_preparation" / "extract_augmented_fast.py",
         args.skip_extract),

        ("2. Extract Static Images",
         base_dir / "data_preparation" / "extract_static_images.py",
         args.skip_images),

        ("3. Prepare Dataset",
         base_dir / "data_preparation" / "prepare_augmented_dataset.py",
         args.skip_prepare),

        ("4. Train Model",
         base_dir / "training" / "train_model.py",
         args.skip_train),
    ]

    print("\nPipeline steps:")
    for name, path, skip in steps:
        status = "[SKIP]" if skip else "[RUN]"
        print(f"  {status} {name}")

    print("\n" + "-" * 70)

    pipeline_start = datetime.now()
    stage_timings = []

    for step_name, script_path, skip in steps:
        if skip:
            print(f"\n[SKIPPED] {step_name}")
            stage_timings.append((step_name, "skipped", None))
            continue

        if not script_path.exists():
            print(f"\n[ERROR] Script not found: {script_path}")
            stage_timings.append((step_name, "failed", timedelta()))
            print_timing_summary(stage_timings)
            return 1

        success, elapsed = run_step(step_name, str(script_path))
        stage_timings.append((step_name, "ok" if success else "failed", elapsed))

        if not success:
            print("\n" + "=" * 70)
            print("  PIPELINE FAILED")
            print("=" * 70)
            print(f"\nFailed at: {step_name}")
            print("Fix the error and run again with --skip flags if needed")
            print_timing_summary(stage_timings)
            return 1

    total_time = datetime.now() - pipeline_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {format_duration(total_time)}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_timing_summary(stage_timings)

    print("\nYour trained model is at: models/alphabet/best_model.pth")
    print("\nTo ship it to the Cloudflare static site:")
    print("  python static-site/tools/export_onnx.py")
    print("  python static-site/tools/build_assets.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
