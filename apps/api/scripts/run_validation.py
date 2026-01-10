#!/usr/bin/env python3
"""
Agentverse Real-World Validation Runner
Reference: agentverse_real_world_validation_playbook.md

This script executes all validation suites and generates a comprehensive report.

Usage:
    python scripts/run_validation.py [--skip-slow] [--output-dir DIR]

Output:
    - validation_report.md: Human-readable report
    - validation_report.json: Machine-readable report
    - reps/: Directory containing all Run Evidence Packs
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.test_center import (
    TestCenter,
    ValidationReport,
    SuiteStatus,
    run_validation,
)
from app.services.rep_service import get_rep_service


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Agentverse Real-World Validation Suites"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow suites (4-6) that require external data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_output",
        help="Output directory for reports and REPs",
    )
    parser.add_argument(
        "--suite",
        type=int,
        nargs="+",
        help="Run specific suites only (e.g., --suite 0 1 2)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗   ██╗███████╗██████╗ ███████╗   ║
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║   ██║██╔════╝██╔══██╗██╔════╝   ║
║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║   ██║█████╗  ██████╔╝███████╗   ║
║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║   ║
║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║    ╚████╔╝ ███████╗██║  ██║███████║   ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝     ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝   ║
║                                                                              ║
║                    Real-World Validation Playbook Runner                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def print_suite_start(suite_num: int, suite_name: str):
    print(f"\n{'='*60}")
    print(f"  Suite {suite_num}: {suite_name}")
    print(f"{'='*60}\n")


def print_test_result(test_name: str, status: str, message: str):
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
    print(f"  {icon} {test_name}: {message}")


def print_suite_result(result):
    print(f"\n  {'─'*50}")
    status_icon = "✅" if result.status == SuiteStatus.PASS else "❌"
    print(f"  {status_icon} Suite Result: {result.status.value}")
    print(f"     Passed: {result.passed} | Failed: {result.failed}")
    print(f"     Duration: {result.duration_seconds:.1f}s")


def print_final_report(report: ValidationReport):
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                       FINAL VALIDATION REPORT                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Overall status
    if report.overall_status == SuiteStatus.PASS:
        print("  ╔═════════════════════════════════════╗")
        print("  ║   ✅ OVERALL STATUS: PASS          ║")
        print("  ╚═════════════════════════════════════╝")
    else:
        print("  ╔═════════════════════════════════════╗")
        print("  ║   ❌ OVERALL STATUS: FAIL          ║")
        print("  ╚═════════════════════════════════════╝")

    print()
    print(f"  Total Tests: {report.total_tests}")
    print(f"  Passed: {report.total_passed}")
    print(f"  Failed: {report.total_failed}")
    print()

    # Suite summary
    print("  Suite Summary:")
    print("  ┌─────────┬────────────────────────────────┬────────┬────────┐")
    print("  │ Suite # │ Name                           │ Status │ Pass/Total │")
    print("  ├─────────┼────────────────────────────────┼────────┼────────┤")

    for suite in report.suites:
        status = "PASS" if suite.status == SuiteStatus.PASS else "FAIL" if suite.status == SuiteStatus.FAIL else "SKIP"
        name = suite.suite_name[:30].ljust(30)
        total = suite.passed + suite.failed
        print(f"  │    {suite.suite_number}    │ {name} │ {status.ljust(6)} │ {suite.passed}/{total}      │")

    print("  └─────────┴────────────────────────────────┴────────┴────────┘")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    ⚠️  {rec}")


async def main():
    args = parse_args()

    print_banner()
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    if args.skip_slow:
        print("⏭️  Skipping slow suites (4-6)")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure REP service to use output directory
    rep_service = get_rep_service(str(output_dir / "reps"))

    # Run validation
    center = TestCenter(rep_service)

    print("Starting validation suites...\n")

    report = await center.run_all_suites(skip_slow=args.skip_slow)

    # Print results for each suite
    for suite in report.suites:
        print_suite_start(suite.suite_number, suite.suite_name)

        for test in suite.tests:
            print_test_result(test.test_name, test.status.value, test.message)

        print_suite_result(suite)

    # Print final report
    print_final_report(report)

    # Generate markdown report
    markdown_report = center.generate_report_markdown(report)
    report_path = output_dir / "validation_report.md"
    with open(report_path, "w") as f:
        f.write(markdown_report)
    print(f"\n📄 Markdown report saved to: {report_path}")

    # Generate JSON report
    json_report_path = output_dir / "validation_report.json"
    with open(json_report_path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)
    print(f"📄 JSON report saved to: {json_report_path}")

    # Summary of REPs
    reps_dir = output_dir / "reps"
    if reps_dir.exists():
        rep_count = len(list(reps_dir.iterdir()))
        print(f"\n📦 {rep_count} Run Evidence Packs generated in: {reps_dir}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")

    # Return exit code based on overall status
    return 0 if report.overall_status == SuiteStatus.PASS else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
