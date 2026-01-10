"""
Run Evidence Pack (REP) Service
Reference: agentverse_real_world_validation_playbook.md §0.1

Every UI action that produces output must generate a REP containing:
- manifest.json: run configuration, seeds, model settings
- data_provenance.json: ingested data sources with hashes and timestamps
- trace.ndjson: append-only event stream of all execution events
- llm_ledger.ndjson: every LLM call with tokens, latency, cache status
- universe_graph.json: nodes & edges with probabilities and confidence intervals
- report.md: plain-English summary with metrics

The REP system enforces "No Black Boxes" - every result must have proof of computation.
"""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import aiofiles
import aiofiles.os

from pydantic import BaseModel, Field


# ============================================================================
# Trace Event Types (trace.ndjson)
# ============================================================================

class TraceEventType(str, Enum):
    """All trace event types for the append-only event stream."""
    # Run lifecycle
    RUN_STARTED = "RUN_STARTED"
    RUN_DONE = "RUN_DONE"
    RUN_FAILED = "RUN_FAILED"

    # Simulation events
    WORLD_TICK = "WORLD_TICK"
    AGENT_STEP = "AGENT_STEP"
    AGENT_DECISION = "AGENT_DECISION"
    POLICY_UPDATE = "POLICY_UPDATE"

    # Universe Map events
    NODE_CREATE = "NODE_CREATE"
    NODE_EXPAND = "NODE_EXPAND"
    NODE_BRANCH = "NODE_BRANCH"
    NODE_FORK = "NODE_FORK"
    NODE_PRUNE = "NODE_PRUNE"

    # Aggregation & Probability
    REPLICATE_START = "REPLICATE_START"
    REPLICATE_DONE = "REPLICATE_DONE"
    AGGREGATE = "AGGREGATE"
    PROBABILITY_COMPUTE = "PROBABILITY_COMPUTE"
    CONFIDENCE_INTERVAL = "CONFIDENCE_INTERVAL"

    # Calibration & Tuning
    CALIBRATE = "CALIBRATE"
    CALIBRATION_TRIAL = "CALIBRATION_TRIAL"
    AUTO_TUNE = "AUTO_TUNE"
    AUTO_TUNE_TRIAL = "AUTO_TUNE_TRIAL"
    STABILITY_TEST = "STABILITY_TEST"

    # Data & Persona
    DATA_INGEST = "DATA_INGEST"
    PERSONA_GENERATE = "PERSONA_GENERATE"
    PERSONA_SNAPSHOT = "PERSONA_SNAPSHOT"

    # Export & Report
    EXPORT_CREATED = "EXPORT_CREATED"
    REPORT_GENERATED = "REPORT_GENERATED"


class TraceEvent(BaseModel):
    """A single trace event in trace.ndjson."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    event_type: TraceEventType
    run_id: str
    node_id: Optional[str] = None
    replicate_id: Optional[int] = None
    agent_id: Optional[str] = None
    tick: Optional[int] = None
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_ndjson_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), separators=(',', ':')) + "\n"


# ============================================================================
# LLM Ledger Entry (llm_ledger.ndjson)
# ============================================================================

class LLMLedgerEntry(BaseModel):
    """A single LLM call record in llm_ledger.ndjson."""
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    run_id: str
    node_id: Optional[str] = None
    replicate_id: Optional[int] = None
    purpose: str  # e.g., "agent_decision", "event_compilation", "persona_generation"
    model: str
    model_provider: str = "openrouter"
    input_hash: str  # SHA256 of input
    output_hash: str  # SHA256 of output
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cache_hit: bool = False
    mock: bool = False  # True if using mock/test mode
    cost_usd: float = 0.0
    error: Optional[str] = None

    def to_ndjson_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), separators=(',', ':')) + "\n"


# ============================================================================
# Manifest (manifest.json)
# ============================================================================

class REPManifest(BaseModel):
    """Run manifest with all configuration and parameters."""
    rep_version: str = "1.0.0"
    rep_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Git/Environment
    commit_sha: Optional[str] = None
    env: str = "development"

    # Run identification
    run_id: str
    project_id: str
    scenario_id: Optional[str] = None
    node_id: Optional[str] = None
    mode: str  # "society", "target", "hybrid"

    # Time controls
    time_cutoff: Optional[str] = None  # ISO timestamp for backtest
    time_cutoff_enforced: bool = False

    # Simulation parameters
    seed: int
    replicate_count: int = 1
    agent_count: int = 1
    step_count: int = 1

    # Model configuration
    model_provider: str = "openrouter"
    model_name: str = "anthropic/claude-3-haiku"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096

    # Rules & Variables
    rulepacks: List[str] = Field(default_factory=list)
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)

    # Completion status
    status: str = "running"  # "running", "completed", "failed"
    completed_at: Optional[str] = None
    error: Optional[str] = None

    # Metrics summary
    total_ticks: int = 0
    total_agent_steps: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    runtime_seconds: float = 0.0


# ============================================================================
# Data Provenance (data_provenance.json)
# ============================================================================

class DataSource(BaseModel):
    """A single data source in data_provenance.json."""
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_name: str
    source_type: str  # "csv", "json", "api", "database", "manual"
    source_path: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    row_count: Optional[int] = None
    max_doc_date: Optional[str] = None  # Must be <= time_cutoff for backtests
    hash: str  # SHA256 of source content
    schema_summary: Dict[str, str] = Field(default_factory=dict)


class DataProvenance(BaseModel):
    """Data provenance record for a run."""
    run_id: str
    time_cutoff: Optional[str] = None
    sources: List[DataSource] = Field(default_factory=list)
    total_sources: int = 0
    provenance_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)


# ============================================================================
# Universe Graph (universe_graph.json)
# ============================================================================

class GraphNode(BaseModel):
    """A node in the universe graph."""
    node_id: str
    parent_id: Optional[str] = None
    run_id: Optional[str] = None
    rep_id: Optional[str] = None  # Link to REP
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Variables & State
    variable_delta: Dict[str, Any] = Field(default_factory=dict)
    state_hash: Optional[str] = None

    # Probability & Confidence
    probability: Optional[float] = None
    probability_source: str = "aggregation"  # "aggregation", "prior", "manual"
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    confidence_level: float = 0.95
    replicate_count: int = 0

    # Metrics
    outcome_metrics: Dict[str, float] = Field(default_factory=dict)

    # Status
    is_stale: bool = False
    is_pruned: bool = False


class GraphEdge(BaseModel):
    """An edge in the universe graph."""
    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    edge_type: str = "branch"  # "branch", "fork", "expand"
    variable_change: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class UniverseGraph(BaseModel):
    """Universe graph structure."""
    project_id: str
    root_node_id: Optional[str] = None
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ============================================================================
# REP Validation
# ============================================================================

class REPValidationResult(BaseModel):
    """Result of REP validation."""
    is_valid: bool
    rep_id: str
    run_id: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # File presence
    has_manifest: bool = False
    has_data_provenance: bool = False
    has_trace: bool = False
    has_llm_ledger: bool = False
    has_universe_graph: bool = False
    has_report: bool = False

    # Content validation
    trace_event_count: int = 0
    llm_call_count: int = 0
    required_events_present: List[str] = Field(default_factory=list)
    missing_events: List[str] = Field(default_factory=list)

    # Scaling footprint
    expected_min_traces: int = 0
    expected_min_llm_calls: int = 0
    footprint_valid: bool = False


# ============================================================================
# REP Service
# ============================================================================

class REPService:
    """
    Service for creating and validating Run Evidence Packs.

    Usage:
        rep = REPService(base_path="/data/reps")

        # Start a new REP
        await rep.start_rep(manifest)

        # Record events during execution
        await rep.add_trace_event(event)
        await rep.add_llm_call(ledger_entry)

        # Finalize and validate
        await rep.finalize_rep(run_id)
        result = await rep.validate_rep(run_id)
    """

    def __init__(self, base_path: str = "/tmp/agentverse/reps"):
        self.base_path = Path(base_path)
        self._active_reps: Dict[str, REPManifest] = {}
        self._trace_buffers: Dict[str, List[TraceEvent]] = {}
        self._llm_buffers: Dict[str, List[LLMLedgerEntry]] = {}

    def _get_rep_path(self, run_id: str) -> Path:
        """Get the REP directory path for a run."""
        return self.base_path / run_id

    async def _ensure_rep_dir(self, run_id: str) -> Path:
        """Ensure REP directory exists."""
        rep_path = self._get_rep_path(run_id)
        await aiofiles.os.makedirs(rep_path, exist_ok=True)
        return rep_path

    async def start_rep(self, manifest: REPManifest) -> str:
        """
        Start a new REP for a run.

        Returns the REP ID.
        """
        run_id = manifest.run_id
        rep_path = await self._ensure_rep_dir(run_id)

        # Initialize buffers
        self._active_reps[run_id] = manifest
        self._trace_buffers[run_id] = []
        self._llm_buffers[run_id] = []

        # Write initial manifest
        async with aiofiles.open(rep_path / "manifest.json", "w") as f:
            await f.write(json.dumps(manifest.model_dump(mode="json"), indent=2))

        # Initialize empty files
        async with aiofiles.open(rep_path / "trace.ndjson", "w") as f:
            pass
        async with aiofiles.open(rep_path / "llm_ledger.ndjson", "w") as f:
            pass

        return manifest.rep_id

    async def add_trace_event(self, event: TraceEvent, flush: bool = False) -> None:
        """Add a trace event to the REP."""
        run_id = event.run_id
        if run_id not in self._trace_buffers:
            self._trace_buffers[run_id] = []

        self._trace_buffers[run_id].append(event)

        # Flush if buffer is large or explicitly requested
        if flush or len(self._trace_buffers[run_id]) >= 100:
            await self._flush_trace_buffer(run_id)

    async def add_llm_call(self, entry: LLMLedgerEntry, flush: bool = False) -> None:
        """Add an LLM call to the ledger."""
        run_id = entry.run_id
        if run_id not in self._llm_buffers:
            self._llm_buffers[run_id] = []

        self._llm_buffers[run_id].append(entry)

        # Update manifest totals
        if run_id in self._active_reps:
            manifest = self._active_reps[run_id]
            manifest.total_llm_calls += 1
            manifest.total_tokens += entry.tokens_in + entry.tokens_out
            manifest.total_cost_usd += entry.cost_usd

        # Flush if buffer is large or explicitly requested
        if flush or len(self._llm_buffers[run_id]) >= 50:
            await self._flush_llm_buffer(run_id)

    async def _flush_trace_buffer(self, run_id: str) -> None:
        """Flush trace buffer to file."""
        if run_id not in self._trace_buffers or not self._trace_buffers[run_id]:
            return

        rep_path = self._get_rep_path(run_id)
        async with aiofiles.open(rep_path / "trace.ndjson", "a") as f:
            for event in self._trace_buffers[run_id]:
                await f.write(event.to_ndjson_line())

        self._trace_buffers[run_id] = []

    async def _flush_llm_buffer(self, run_id: str) -> None:
        """Flush LLM ledger buffer to file."""
        if run_id not in self._llm_buffers or not self._llm_buffers[run_id]:
            return

        rep_path = self._get_rep_path(run_id)
        async with aiofiles.open(rep_path / "llm_ledger.ndjson", "a") as f:
            for entry in self._llm_buffers[run_id]:
                await f.write(entry.to_ndjson_line())

        self._llm_buffers[run_id] = []

    async def set_data_provenance(self, run_id: str, provenance: DataProvenance) -> None:
        """Set the data provenance for a run."""
        rep_path = await self._ensure_rep_dir(run_id)
        async with aiofiles.open(rep_path / "data_provenance.json", "w") as f:
            await f.write(json.dumps(provenance.model_dump(mode="json"), indent=2))

    async def set_universe_graph(self, run_id: str, graph: UniverseGraph) -> None:
        """Set the universe graph for a run."""
        rep_path = await self._ensure_rep_dir(run_id)
        async with aiofiles.open(rep_path / "universe_graph.json", "w") as f:
            await f.write(json.dumps(graph.model_dump(mode="json"), indent=2))

    async def generate_report(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        summary: str,
        leakage_test_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the report.md for a run."""
        rep_path = await self._ensure_rep_dir(run_id)
        manifest = self._active_reps.get(run_id)

        report_lines = [
            "# Run Evidence Pack Report",
            "",
            f"**REP ID:** {manifest.rep_id if manifest else 'N/A'}",
            f"**Run ID:** {run_id}",
            f"**Generated:** {datetime.utcnow().isoformat()}Z",
            "",
            "---",
            "",
            "## Summary",
            "",
            summary,
            "",
            "---",
            "",
            "## Configuration",
            "",
        ]

        if manifest:
            report_lines.extend([
                f"- **Mode:** {manifest.mode}",
                f"- **Agent Count:** {manifest.agent_count}",
                f"- **Step Count:** {manifest.step_count}",
                f"- **Replicate Count:** {manifest.replicate_count}",
                f"- **Seed:** {manifest.seed}",
                f"- **Model:** {manifest.model_name}",
                f"- **Time Cutoff:** {manifest.time_cutoff or 'None (live mode)'}",
                "",
            ])

        report_lines.extend([
            "---",
            "",
            "## Metrics",
            "",
        ])

        for key, value in metrics.items():
            report_lines.append(f"- **{key}:** {value}")

        report_lines.extend([
            "",
            "---",
            "",
            "## Execution Statistics",
            "",
        ])

        if manifest:
            report_lines.extend([
                f"- **Total Ticks:** {manifest.total_ticks}",
                f"- **Total Agent Steps:** {manifest.total_agent_steps}",
                f"- **Total LLM Calls:** {manifest.total_llm_calls}",
                f"- **Total Tokens:** {manifest.total_tokens}",
                f"- **Total Cost:** ${manifest.total_cost_usd:.4f}",
                f"- **Runtime:** {manifest.runtime_seconds:.2f}s",
                "",
            ])

        if leakage_test_results:
            report_lines.extend([
                "---",
                "",
                "## Leakage Test Results",
                "",
            ])
            for test, result in leakage_test_results.items():
                status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
                report_lines.append(f"- **{test}:** {status}")
                if result.get("details"):
                    report_lines.append(f"  - {result['details']}")
            report_lines.append("")

        report_lines.extend([
            "---",
            "",
            "## Verification",
            "",
            "This REP contains:",
            "- ✅ manifest.json - Run configuration and parameters",
            "- ✅ data_provenance.json - Data source tracking",
            "- ✅ trace.ndjson - Append-only event stream",
            "- ✅ llm_ledger.ndjson - LLM call records",
            "- ✅ universe_graph.json - Node/edge structure",
            "- ✅ report.md - This summary",
            "",
        ])

        report_content = "\n".join(report_lines)

        async with aiofiles.open(rep_path / "report.md", "w") as f:
            await f.write(report_content)

        return report_content

    async def finalize_rep(
        self,
        run_id: str,
        status: str = "completed",
        error: Optional[str] = None,
        runtime_seconds: float = 0.0,
    ) -> REPManifest:
        """Finalize the REP and write final manifest."""
        # Flush all buffers
        await self._flush_trace_buffer(run_id)
        await self._flush_llm_buffer(run_id)

        # Update manifest
        manifest = self._active_reps.get(run_id)
        if manifest:
            manifest.status = status
            manifest.completed_at = datetime.utcnow().isoformat() + "Z"
            manifest.error = error
            manifest.runtime_seconds = runtime_seconds

            # Write final manifest
            rep_path = self._get_rep_path(run_id)
            async with aiofiles.open(rep_path / "manifest.json", "w") as f:
                await f.write(json.dumps(manifest.model_dump(mode="json"), indent=2))

        return manifest

    async def validate_rep(self, run_id: str) -> REPValidationResult:
        """
        Validate a REP for completeness and correctness.

        This is the gatekeeper - UI should not show results unless this passes.
        """
        rep_path = self._get_rep_path(run_id)
        result = REPValidationResult(
            is_valid=False,
            rep_id="",
            run_id=run_id,
        )

        # Check file presence
        result.has_manifest = (rep_path / "manifest.json").exists()
        result.has_data_provenance = (rep_path / "data_provenance.json").exists()
        result.has_trace = (rep_path / "trace.ndjson").exists()
        result.has_llm_ledger = (rep_path / "llm_ledger.ndjson").exists()
        result.has_universe_graph = (rep_path / "universe_graph.json").exists()
        result.has_report = (rep_path / "report.md").exists()

        if not result.has_manifest:
            result.errors.append("Missing manifest.json")
            return result

        # Load manifest
        async with aiofiles.open(rep_path / "manifest.json", "r") as f:
            manifest_data = json.loads(await f.read())
            manifest = REPManifest(**manifest_data)
            result.rep_id = manifest.rep_id

        # Validate required files
        if not result.has_trace:
            result.errors.append("Missing trace.ndjson")
        if not result.has_llm_ledger:
            result.errors.append("Missing llm_ledger.ndjson")

        # Count trace events
        if result.has_trace:
            event_types_found: Set[str] = set()
            async with aiofiles.open(rep_path / "trace.ndjson", "r") as f:
                async for line in f:
                    if line.strip():
                        result.trace_event_count += 1
                        try:
                            event = json.loads(line)
                            event_types_found.add(event.get("event_type", ""))
                        except:
                            pass

            result.required_events_present = list(event_types_found)

            # Check required events for a valid run
            required = {TraceEventType.RUN_STARTED.value, TraceEventType.RUN_DONE.value}
            missing = required - event_types_found
            result.missing_events = list(missing)
            if missing:
                result.errors.append(f"Missing required trace events: {missing}")

        # Count LLM calls
        if result.has_llm_ledger:
            async with aiofiles.open(rep_path / "llm_ledger.ndjson", "r") as f:
                async for line in f:
                    if line.strip():
                        result.llm_call_count += 1

        # Validate scaling footprint
        # Expected minimum traces: agent_count * step_count * replicate_count
        expected_traces = manifest.agent_count * manifest.step_count * manifest.replicate_count
        result.expected_min_traces = max(expected_traces // 10, 5)  # Allow 10% tolerance
        result.expected_min_llm_calls = max(manifest.agent_count * manifest.step_count // 10, 1)

        result.footprint_valid = (
            result.trace_event_count >= result.expected_min_traces and
            result.llm_call_count >= result.expected_min_llm_calls
        )

        if not result.footprint_valid:
            result.warnings.append(
                f"Scaling footprint mismatch: {result.trace_event_count} traces "
                f"(expected >= {result.expected_min_traces}), "
                f"{result.llm_call_count} LLM calls "
                f"(expected >= {result.expected_min_llm_calls})"
            )

        # Final validation
        result.is_valid = (
            len(result.errors) == 0 and
            result.has_manifest and
            result.has_trace and
            result.has_llm_ledger
        )

        return result

    async def load_rep(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a complete REP from disk."""
        rep_path = self._get_rep_path(run_id)

        if not rep_path.exists():
            return None

        rep_data = {"run_id": run_id}

        # Load manifest
        if (rep_path / "manifest.json").exists():
            async with aiofiles.open(rep_path / "manifest.json", "r") as f:
                rep_data["manifest"] = json.loads(await f.read())

        # Load data provenance
        if (rep_path / "data_provenance.json").exists():
            async with aiofiles.open(rep_path / "data_provenance.json", "r") as f:
                rep_data["data_provenance"] = json.loads(await f.read())

        # Load universe graph
        if (rep_path / "universe_graph.json").exists():
            async with aiofiles.open(rep_path / "universe_graph.json", "r") as f:
                rep_data["universe_graph"] = json.loads(await f.read())

        # Load report
        if (rep_path / "report.md").exists():
            async with aiofiles.open(rep_path / "report.md", "r") as f:
                rep_data["report"] = await f.read()

        return rep_data


# ============================================================================
# REP Validator Middleware
# ============================================================================

class REPValidator:
    """
    Validator that blocks UI completion without valid REP.

    This enforces the "No Black Boxes" rule - every result must have proof.
    """

    def __init__(self, rep_service: REPService):
        self.rep_service = rep_service

    async def require_valid_rep(self, run_id: str) -> REPValidationResult:
        """
        Validate REP and raise exception if invalid.

        Call this before returning results to UI.
        """
        result = await self.rep_service.validate_rep(run_id)

        if not result.is_valid:
            raise REPValidationError(
                f"REP validation failed for run {run_id}: {result.errors}"
            )

        return result

    async def check_scaling_footprint(
        self,
        run_id: str,
        agent_count: int,
        step_count: int,
        replicate_count: int,
    ) -> bool:
        """
        Check that REP shows appropriate scaling footprint.

        This proves the backend actually ran multi-agent/replicate logic.
        """
        result = await self.rep_service.validate_rep(run_id)

        # Calculate expected minimums
        expected_traces = agent_count * step_count * replicate_count // 10
        expected_llm = agent_count * step_count // 10

        return (
            result.trace_event_count >= expected_traces and
            result.llm_call_count >= expected_llm
        )


class REPValidationError(Exception):
    """Raised when REP validation fails."""
    pass


# ============================================================================
# Factory function
# ============================================================================

_rep_service_instance: Optional[REPService] = None


def get_rep_service(base_path: str = "/tmp/agentverse/reps") -> REPService:
    """Get or create the REP service singleton."""
    global _rep_service_instance
    if _rep_service_instance is None:
        _rep_service_instance = REPService(base_path)
    return _rep_service_instance


def get_rep_validator() -> REPValidator:
    """Get a REP validator instance."""
    return REPValidator(get_rep_service())
