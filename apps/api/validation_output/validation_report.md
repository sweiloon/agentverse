# Agentverse Validation Report

**Generated:** 2026-01-10T10:55:02.803501Z
**Report ID:** b12a2ae5-2115-4c08-afea-336b250ac3b8
**Overall Status:** ✅ PASS

---

## Summary

- **Total Tests:** 15
- **Passed:** 15
- **Failed:** 0

---

## Suite 0: Smoke & Observability

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 4/4 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Create Run with REP | ✅ | 6ms | Successfully created run with REP |
| REP Files Present | ✅ | 1ms | All required files present |
| Trace Events | ✅ | 1ms | All required events present |
| LLM Ledger | ✅ | 1ms | 50 LLM calls recorded |

---

## Suite 1: Scaling Proof

**Status:** ✅ PASS
**Duration:** 4.6s
**Summary:** 3/3 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Run A (Small) | ✅ | 21ms | 339 traces, 300 LLM calls |
| Run B (Large) | ✅ | 4263ms | 60323 traces, 60000 LLM calls |
| Scaling Comparison | ✅ | 271ms | Trace ratio: 177.9x, LLM ratio: 200.0x |

---

## Suite 2: Universe Map Correctness

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 2/2 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Node Expansion | ✅ | 6ms | 3/3 child REPs valid |
| Probability Aggregation | ✅ | 4ms | Mean: 0.547, CI: [0.455, 0.638] |

---

## Suite 3: Calibration + Auto-Tune

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 2/2 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Calibration Trials | ✅ | 2ms | Calibration trials logged |
| Auto-Tune Trials | ✅ | 2ms | Auto-tune trials logged |

---

## Suite 4: Society Mode Backtest

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 2/2 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Time Cutoff Enforcement | ✅ | 0ms | Filtered 1 post-cutoff documents |
| Leakage Canary | ✅ | 0ms | Post-cutoff question properly refused |

---

## Suite 5: Target Mode Backtest

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 1/1 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Per-Persona Decision Traces | ✅ | 11ms | 100 persona decisions logged |

---

## Suite 6: Hybrid Mode Backtest

**Status:** ✅ PASS
**Duration:** 0.0s
**Summary:** 1/1 tests passed

### Tests

| Test | Status | Duration | Message |
|------|--------|----------|---------|
| Hybrid Submodels | ✅ | 2ms | Separate submodel policies logged |

---
