"""
API v1 Router
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    projects,
    scenarios,
    simulations,
    users,
    data_sources,
    personas,
    products,
    validation,
    ai_generation,
    focus_groups,
    organizations,
    invitations,
    marketplace,
    world,
    predictions,
    # Spec-compliant endpoints (project.md §6)
    runs,
    nodes,
    telemetry,
    project_specs,
    event_scripts,
    ask,
    # Target Mode (project.md §11 Phase 5)
    target_mode,
    # Calibration & Reliability (project.md §11 Phase 7)
    calibration,
    # 2D Replay (project.md §11 Phase 8)
    replay,
    # Export Controls (project.md §11 Phase 9)
    exports,
    # Privacy & Compliance (project.md §11 Phase 9)
    privacy,
    # LLM Router Admin (GAPS.md GAP-P0-001)
    llm_admin,
    # Audit Log Admin (GAPS.md GAP-P0-006)
    audit_admin,
    # Evidence Pack API (verification_checklist_v2.md §1)
    evidence,
    # Knowledge Graph / Universe Ops (STEP 9)
    universe_graph,
    # Governance, Cost Controls, Safety (STEP 10)
    governance,
    # Validation Center (Real-World Validation Playbook)
    validation_center,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(scenarios.router, prefix="/scenarios", tags=["Scenarios"])
api_router.include_router(simulations.router, prefix="/simulations", tags=["Simulations"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictive Simulations"])
api_router.include_router(data_sources.router, prefix="/data-sources", tags=["Data Sources"])
api_router.include_router(personas.router, prefix="/personas", tags=["Personas"])
api_router.include_router(products.router, prefix="/products", tags=["Products"])
api_router.include_router(validation.router, prefix="/validation", tags=["Validation & Accuracy"])
api_router.include_router(ai_generation.router, prefix="/ai", tags=["AI Content Generation"])
api_router.include_router(focus_groups.router, prefix="/focus-groups", tags=["Virtual Focus Groups"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(invitations.router, prefix="/invitations", tags=["Invitations"])
api_router.include_router(marketplace.router, prefix="/marketplace", tags=["Marketplace"])
api_router.include_router(world.router, prefix="/world", tags=["Vi World"])

# Spec-compliant endpoints (project.md §6)
api_router.include_router(runs.router, prefix="/runs", tags=["Simulation Runs"])
api_router.include_router(nodes.router, prefix="/nodes", tags=["Universe Map"])
api_router.include_router(telemetry.router, prefix="/telemetry", tags=["Telemetry"])
api_router.include_router(project_specs.router, prefix="/project-specs", tags=["Project Specs"])
api_router.include_router(event_scripts.router, tags=["Event Scripts"])
api_router.include_router(ask.router, prefix="/ask", tags=["Ask - Event Compiler"])
api_router.include_router(target_mode.router, prefix="/target", tags=["Target Mode"])

# Calibration & Reliability (project.md §11 Phase 7)
api_router.include_router(calibration.router, prefix="/calibration", tags=["Calibration & Reliability"])

# 2D Replay (project.md §11 Phase 8) - READ-ONLY per C3
api_router.include_router(replay.router, prefix="/replay", tags=["2D Replay"])

# Export Controls (project.md §11 Phase 9)
api_router.include_router(exports.router, prefix="/exports", tags=["Exports"])

# Privacy & Compliance (project.md §11 Phase 9)
api_router.include_router(privacy.router, prefix="/privacy", tags=["Privacy & Compliance"])

# LLM Router Admin (GAPS.md GAP-P0-001) - Admin only
api_router.include_router(llm_admin.router, prefix="/admin/llm", tags=["LLM Admin"])

# Audit Log Admin (GAPS.md GAP-P0-006) - Admin only
api_router.include_router(audit_admin.router, prefix="/admin", tags=["Audit Admin"])

# Evidence Pack API (verification_checklist_v2.md §1)
api_router.include_router(evidence.router, prefix="/evidence", tags=["Evidence & Verification"])

# Knowledge Graph / Universe Ops (STEP 9)
api_router.include_router(universe_graph.router, prefix="/universe-graph", tags=["Knowledge Graph / Universe Ops"])

# Governance, Cost Controls, Safety (STEP 10)
api_router.include_router(governance.router, prefix="/governance", tags=["Governance & Cost Controls"])

# Validation Center (Real-World Validation Playbook)
api_router.include_router(validation_center.router, prefix="/validation-center", tags=["Validation Center"])
