"""
Validation Center API Endpoints
Reference: agentverse_real_world_validation_playbook.md

Provides endpoints to:
- Run validation suites
- Get REP validation status
- View test results
- Block UI completion without valid REP
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.services.rep_service import (
    REPService,
    REPValidator,
    REPValidationResult,
    REPValidationError,
    REPManifest,
    get_rep_service,
    get_rep_validator,
)
from app.services.test_center import (
    TestCenter,
    SuiteResult,
    TestResult,
    ValidationReport,
    SuiteStatus,
)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class RunValidationRequest(BaseModel):
    """Request to run validation suites."""
    skip_slow: bool = False
    suites: Optional[List[int]] = None  # None = all suites


class RunValidationResponse(BaseModel):
    """Response from validation run."""
    report_id: str
    overall_status: str
    total_tests: int
    total_passed: int
    total_failed: int
    suites: List[Dict[str, Any]]
    recommendations: List[str]


class REPStatusRequest(BaseModel):
    """Request to check REP status."""
    run_id: str


class REPStatusResponse(BaseModel):
    """Response with REP validation status."""
    run_id: str
    rep_id: Optional[str] = None
    is_valid: bool
    can_show_ui: bool
    errors: List[str]
    warnings: List[str]
    file_status: Dict[str, bool]
    trace_event_count: int
    llm_call_count: int
    footprint_valid: bool


class RequireREPRequest(BaseModel):
    """Request to require REP for UI."""
    run_id: str


class RequireREPResponse(BaseModel):
    """Response from REP requirement check."""
    run_id: str
    allowed: bool
    rep_valid: bool
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/run-validation", response_model=RunValidationResponse)
async def run_validation(request: RunValidationRequest) -> RunValidationResponse:
    """
    Run validation suites.

    Returns comprehensive validation report with PASS/FAIL status for each suite.
    """
    center = TestCenter()
    report = await center.run_all_suites(skip_slow=request.skip_slow)

    return RunValidationResponse(
        report_id=report.report_id,
        overall_status=report.overall_status.value,
        total_tests=report.total_tests,
        total_passed=report.total_passed,
        total_failed=report.total_failed,
        suites=[
            {
                "suite_number": s.suite_number,
                "suite_name": s.suite_name,
                "status": s.status.value,
                "duration_seconds": s.duration_seconds,
                "passed": s.passed,
                "failed": s.failed,
                "summary": s.summary,
                "tests": [
                    {
                        "test_name": t.test_name,
                        "status": t.status.value,
                        "duration_ms": t.duration_ms,
                        "message": t.message,
                    }
                    for t in s.tests
                ],
            }
            for s in report.suites
        ],
        recommendations=report.recommendations,
    )


@router.post("/rep-status", response_model=REPStatusResponse)
async def get_rep_status(request: REPStatusRequest) -> REPStatusResponse:
    """
    Get REP validation status for a run.

    Use this to check if a run has a valid REP before showing results to UI.
    """
    rep_service = get_rep_service()
    result = await rep_service.validate_rep(request.run_id)

    return REPStatusResponse(
        run_id=request.run_id,
        rep_id=result.rep_id if result.rep_id else None,
        is_valid=result.is_valid,
        can_show_ui=result.is_valid and result.footprint_valid,
        errors=result.errors,
        warnings=result.warnings,
        file_status={
            "manifest": result.has_manifest,
            "data_provenance": result.has_data_provenance,
            "trace": result.has_trace,
            "llm_ledger": result.has_llm_ledger,
            "universe_graph": result.has_universe_graph,
            "report": result.has_report,
        },
        trace_event_count=result.trace_event_count,
        llm_call_count=result.llm_call_count,
        footprint_valid=result.footprint_valid,
    )


@router.post("/require-rep", response_model=RequireREPResponse)
async def require_rep_for_ui(request: RequireREPRequest) -> RequireREPResponse:
    """
    Enforce REP requirement before showing UI results.

    This is the gatekeeper - call this before displaying any run results to users.
    Returns allowed=False if REP is invalid or missing.
    """
    validator = get_rep_validator()

    try:
        result = await validator.require_valid_rep(request.run_id)
        return RequireREPResponse(
            run_id=request.run_id,
            allowed=True,
            rep_valid=True,
            message="REP valid - UI display allowed",
        )
    except REPValidationError as e:
        return RequireREPResponse(
            run_id=request.run_id,
            allowed=False,
            rep_valid=False,
            message=str(e),
        )


@router.get("/validation-report/{report_id}")
async def get_validation_report(report_id: str) -> Dict[str, Any]:
    """
    Get a previously generated validation report.

    Note: Reports are ephemeral in this implementation.
    For production, store reports in database.
    """
    # In production, this would fetch from database
    raise HTTPException(
        status_code=404,
        detail="Report not found. Run /run-validation to generate a new report.",
    )


@router.get("/rep/{run_id}")
async def get_rep(run_id: str) -> Dict[str, Any]:
    """
    Get the complete REP for a run.

    Returns all REP artifacts (manifest, traces summary, etc.)
    """
    rep_service = get_rep_service()
    rep = await rep_service.load_rep(run_id)

    if not rep:
        raise HTTPException(
            status_code=404,
            detail=f"REP not found for run {run_id}",
        )

    return rep


@router.get("/suites")
async def list_suites() -> List[Dict[str, Any]]:
    """
    List available validation suites.
    """
    return [
        {
            "number": 0,
            "name": "Smoke & Observability",
            "description": "Verify platform can produce REPs end-to-end",
            "duration_estimate": "30 minutes",
        },
        {
            "number": 1,
            "name": "Scaling Proof",
            "description": "Prove backend runs multi-agent/replicate logic (not single LLM call)",
            "duration_estimate": "1 hour",
        },
        {
            "number": 2,
            "name": "Universe Map Correctness",
            "description": "Verify Universe Map is derived from simulation + probability aggregation",
            "duration_estimate": "2 hours",
        },
        {
            "number": 3,
            "name": "Calibration + Auto-Tune",
            "description": "Verify calibration updates parameters and improves metrics",
            "duration_estimate": "3-6 hours",
        },
        {
            "number": 4,
            "name": "Society Mode Backtest",
            "description": "2024 US Presidential Election prediction with time cutoff",
            "duration_estimate": "4+ hours",
            "requires_data": True,
        },
        {
            "number": 5,
            "name": "Target Mode Backtest",
            "description": "UCI Bank Marketing dataset - individual decision prediction",
            "duration_estimate": "2+ hours",
            "requires_data": True,
        },
        {
            "number": 6,
            "name": "Hybrid Mode Backtest",
            "description": "E-commerce A/B Testing - crowd + segment modeling",
            "duration_estimate": "2+ hours",
            "requires_data": True,
        },
    ]


@router.get("/health")
async def validation_health() -> Dict[str, Any]:
    """
    Health check for validation system.
    """
    rep_service = get_rep_service()

    return {
        "status": "healthy",
        "rep_service": "active",
        "base_path": str(rep_service.base_path),
        "active_reps": len(rep_service._active_reps),
    }
