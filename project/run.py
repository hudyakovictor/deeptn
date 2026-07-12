"""
DEEPUTIN Pipeline Runner
========================
Main entry point for the forensic analysis pipeline.

Stages:
  S1: Extraction — 3DDFA reconstruction, face mask, pose estimation, geometry + texture metrics
  S2: Identity — calibration reference building
  S3: Compare — pairwise evidence computation
  S4: Verdict — Bayesian forensic verdicts
  S5: Report — markdown report + HTML timeline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# ── Bootstrap imports ──
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deeputin.shared.logging import setup_logger, setup_file_logging
    from deeputin.shared.config_validation import validate_config
    from deeputin.shared.progress import PipelineProgress
    from deeputin.shared.schemas import PipelineDataset
    from deeputin.shared.utils import ensure_dir, load_yaml
    from deeputin.s1_extraction import ExtractionEngine
    from deeputin.s2_identity import CalibrationEngine
    from deeputin.s3_compare import CompareEngine
    from deeputin.s4_verdict import VerdictEngine
    from deeputin.s5_report import ReportEngine
else:
    from .shared.logging import setup_logger, setup_file_logging
    from .shared.config_validation import validate_config
    from .shared.progress import PipelineProgress
    from .shared.schemas import PipelineDataset
    from .shared.utils import ensure_dir, load_yaml
    from .s1_extraction import ExtractionEngine
    from .s2_identity import CalibrationEngine
    from .s3_compare import CompareEngine
    from .s4_verdict import VerdictEngine
    from .s5_report import ReportEngine

log = setup_logger("pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DPTN_DATA_ROOT", PROJECT_ROOT / "data"))

DEFAULT_MAIN_INPUT = DATA_ROOT / "photo" / "all"
DEFAULT_CALIBRATION_INPUT = DATA_ROOT / "photo" / "calibration"
DEFAULT_MAIN_OUTPUT = DATA_ROOT / "storage" / "main"
DEFAULT_CALIBRATION_OUTPUT = DATA_ROOT / "storage" / "calibration"
DEFAULT_STAGES = ("s1", "s2", "s3", "s4", "s5")

STAGE_NAMES = {
    "s1": "S1: Extract + Metrics",
    "s2": "S2: Identity",
    "s3": "S3: Compare",
    "s4": "S4: Verdict",
    "s5": "S5: Report",
}


class PipelineRunner:
    """
    Orchestrates the full DEEPUTIN forensic pipeline.
    
    Features:
    - Color-coded console logging (white=OK, yellow=warning, red=error)
    - File logging (all messages to deeputin_*.log)
    - Progress bars for each stage
    - Soft validation (warnings instead of crashes)
    - Clean JSON output (no debug/internal fields)
    """

    def __init__(
        self,
        main_input: str | Path = DEFAULT_MAIN_INPUT,
        calibration_input: str | Path = DEFAULT_CALIBRATION_INPUT,
        main_output: str | Path = DEFAULT_MAIN_OUTPUT,
        calibration_output: str | Path = DEFAULT_CALIBRATION_OUTPUT,
        config_path: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.main_input = Path(main_input)
        self.calibration_input = Path(calibration_input)
        self.main_output = ensure_dir(main_output)
        self.calibration_output = ensure_dir(calibration_output)
        self.config = load_yaml(config_path, default={}) if config_path else {}
        self.limit = limit
        self._total_errors = 0
        self._total_warnings = 0

    def run(self, stages: Iterable[str] = DEFAULT_STAGES) -> dict[str, object]:
        """Run the pipeline for the specified stages."""
        stages = tuple(s for s in stages if s in STAGE_NAMES)
        if not stages:
            log.warning("No valid stages specified")
            return {}

        # Set up file logging
        log_file = setup_file_logging(self.main_output)
        log.info(f"Log file: {log_file}")

        # Print header
        log.info("=" * 60)
        log.info("DEEPUTIN Forensic Pipeline")
        log.info("=" * 60)
        log.info(f"Stages: {', '.join(STAGE_NAMES.get(s, s) for s in stages)}")
        log.info(f"Main input: {self.main_input}")
        log.info(f"Main output: {self.main_output}")
        if self.limit:
            log.info(f"Limit: {self.limit} photos per dataset")
        log.info("")

        result: dict[str, object] = {}
        pipeline_start = time.time()

        # Count photos for progress tracking
        main_photos = self._count_photos(self.main_output) if "s1" not in stages else self._count_input_photos(self.main_input)
        cal_photos = self._count_photos(self.calibration_output) if "s1" not in stages else self._count_input_photos(self.calibration_input)

        # ── S1: Extraction + Metrics (inline) ──
        stage1_main = stage1_cal = None
        stage1_main_records = []
        stage1_cal_records = []
        if "s1" in stages:
            log.info(f"── {STAGE_NAMES['s1']} ──")

            log.info(f"Processing main dataset ({main_photos} photos)...")
            stage1_main_records, main_err, main_warn = self._run_stage1(self.main_input, self.main_output, PipelineDataset.MAIN)
            self._total_errors += main_err

            log.info(f"Processing calibration dataset ({cal_photos} photos)...")
            stage1_cal_records, cal_err, cal_warn = self._run_stage1(self.calibration_input, self.calibration_output, PipelineDataset.CALIBRATION)
            self._total_errors += cal_err

            stage1_main = stage1_main_records
            stage1_cal = stage1_cal_records

            result["stage1"] = {
                "main_count": len(stage1_main_records),
                "calibration_count": len(stage1_cal_records),
                "main_errors": main_err,
                "calibration_errors": cal_err,
            }
            log.success(f"S1 complete: {len(stage1_main_records)} main + {len(stage1_cal_records)} calibration photos")

        # ── S2: Identity ──
        if "s2" in stages:
            log.info(f"── {STAGE_NAMES['s2']} ──")
            try:
                calibration_engine = CalibrationEngine(config=self.config.get("s2", {}))
                reference = calibration_engine.build_reference(self.calibration_output)
                if reference is not None:
                    calibration_engine.save_reference(reference, self.calibration_output / "calibration_reference.json")
                    result["stage2_reference"] = reference.model_dump()
                    calibration_engine.annotate_main_dataset(self.main_output, reference)
                    log.success(f"S2 complete: calibration reference built")
                else:
                    log.warning("S2: calibration reference could not be built")
            except Exception as e:
                log.error(f"S2 failed: {e}")
                self._total_errors += 1

        # ── S3: Compare ──
        if "s3" in stages:
            log.info(f"── {STAGE_NAMES['s3']} ──")
            try:
                compare_engine = CompareEngine(config=self.config.get("s3", {}))
                pairs = compare_engine.build_pairwise_evidence(
                    self.main_output,
                    reference_path=self.calibration_output / "calibration_reference.json",
                )
                result["stage3_pairs"] = len(pairs)
                log.success(f"S3 complete: {len(pairs)} pairs computed")
            except Exception as e:
                log.error(f"S3 failed: {e}")
                self._total_errors += 1

        # ── S4: Verdict ──
        if "s4" in stages:
            log.info(f"── {STAGE_NAMES['s4']} ──")
            try:
                verdict_engine = VerdictEngine(config=self.config.get("s4", {}))
                verdicts, timeline = verdict_engine.build_verdicts(self.main_output)
                result["stage4"] = {
                    "verdict_count": len(verdicts),
                    "timeline_count": len(timeline),
                }
                log.success(f"S4 complete: {len(verdicts)} verdicts, {len(timeline)} timeline entries")
            except Exception as e:
                log.error(f"S4 failed: {e}")
                self._total_errors += 1

        # ── S5: Report ──
        if "s5" in stages:
            log.info(f"── {STAGE_NAMES['s5']} ──")
            try:
                report_engine = ReportEngine(config=self.config.get("s5", {}))
                report = report_engine.build_report(self.main_output)
                result["stage5"] = report.model_dump()
                log.success(f"S5 complete: report generated")
            except Exception as e:
                log.error(f"S6 failed: {e}")
                self._total_errors += 1

        # ── Summary ──
        elapsed = time.time() - pipeline_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        if len(stage1_main_records) == 0 and "s1" in stages:
            log.error("FATAL: No main photos were successfully processed")
            self._total_errors += 1

        log.info("")
        log.info("=" * 60)
        if self._total_errors > 0:
            log.error(f"Pipeline completed with {self._total_errors} errors in {mins}m{secs:02d}s")
        else:
            log.success(f"Pipeline completed successfully in {mins}m{secs:02d}s")
        log.info("=" * 60)

        result["total_errors"] = self._total_errors
        result["elapsed_seconds"] = elapsed
        return result

    def _run_stage1(self, input_dir: Path, output_dir: Path, dataset: PipelineDataset):
        engine = ExtractionEngine(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset=dataset,
            limit=self.limit,
            config=self.config,
        )
        return engine.run()

    def _count_photos(self, directory: Path) -> int:
        """Count processed photo directories."""
        if not directory.exists():
            return 0
        return sum(1 for d in directory.iterdir() if d.is_dir() and not d.name.startswith("."))

    def _count_input_photos(self, directory: Path) -> int:
        """Count input photo files."""
        if not directory.exists():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deeputin",
        description="DEEPUTIN forensic pipeline — facial analysis for identity verification",
    )

    parser.add_argument(
        "--stages", nargs="*", default=list(DEFAULT_STAGES),
        help="Stages to run: s1 s2 s3 s4 s5 (default: all)",
    )
    parser.add_argument("--input-main", default=str(DEFAULT_MAIN_INPUT), help="Main photo directory")
    parser.add_argument("--input-calibration", default=str(DEFAULT_CALIBRATION_INPUT), help="Calibration photo directory")
    parser.add_argument("--output-main", default=str(DEFAULT_MAIN_OUTPUT), help="Main output directory")
    parser.add_argument("--output-calibration", default=str(DEFAULT_CALIBRATION_OUTPUT), help="Calibration output directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--limit", type=int, default=None, help="Limit photos per dataset (for testing)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config:
        config = load_yaml(args.config, default={})
        errors = validate_config(config)
        if errors:
            for err in errors:
                log.error(f"Config error: {err}")
            return 1

    runner = PipelineRunner(
        main_input=args.input_main,
        calibration_input=args.input_calibration,
        main_output=args.output_main,
        calibration_output=args.output_calibration,
        config_path=args.config,
        limit=args.limit,
    )
    result = runner.run(args.stages)
    return 1 if runner._total_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
