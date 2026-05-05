# STATUS.md — Honest Implementation Status

> **Last updated:** May 2026  
> This document replaces the aspirational `PHASE*.md` / `REVOLUTIONARY_ACHIEVEMENTS.md` files
> with an evidence-based assessment of what actually works, what is partially implemented,
> and what remains to be done.

---

## What Actually Works (Verified)

| Component | Status | Notes |
|---|---|---|
| GnuCash **SQLite** backend | ✅ Working | Full read access — accounts, transactions, splits, spending by category |
| GnuCash **XML** backend | ✅ Working | Parses plain and gzip-compressed GnuCash XML files |
| Transaction **regex categorization** | ✅ Working | ~12 categories, merchant cache |
| Transaction **ML categorization** | ✅ Working | scikit-learn TF-IDF + LogisticRegression; trains automatically once ≥5 samples/class |
| Statistical **anomaly detection** | ✅ Working | Z-score + IQR on transaction amounts; behavioral & temporal anomalies |
| NL query **intent parsing** | ✅ Working | Regex-based intent recognition for 9 query types |
| NL query **execution** | ✅ Working | BALANCE, SPENDING, INCOME, SUMMARY, SEARCH queries execute real SQLite/XML queries |
| ElizaOS bridge (`AtomSpaceProvider`) | ✅ Working | Connects to live atomspace-restful when available; falls back to in-process dict |
| ElizaOS bridge (`CogServerAction`) | ✅ Working | Real HTTP POSTs to CogServer; falls back gracefully |
| AtomSpace core (local) | ✅ Working | In-process atom/link store with optional server sync |
| AtomSpace **server sync** | ✅ Working (when server up) | Persists atoms to atomspace-restful REST API |
| AsyncIO framework | ✅ Working | All bridges use proper async/await |
| Unit test suite (`tests/test_real_logic.py`) | ✅ Working | Runs fully offline; ~30 tests covering all real logic |
| CI workflow (`python-ci.yml`) | ✅ Added | Runs unit tests on every push/PR |

---

## Partially Implemented

| Component | Status | Gap |
|---|---|---|
| PLN reasoning (`PLNReasoner`) | 🟨 Rule-based fallback | No real PLN server; uses heuristic conclusions |
| Financial forecasting | 🟨 Simple trend models | No trained ML forecasting model |
| Risk assessment | 🟨 Placeholder scores | Liquidity/volatility methods return static values |
| Microservices (FastAPI) | 🟨 Skeleton | Endpoints defined but no business logic |
| Embodiment layer (ROS/Unity) | 🟨 Interface stubs | Class definitions exist; no real robot/game integration |
| AtomSpace pattern matching | 🟨 Basic | Server-side queries use name filter only; no full Atomese patterns |

---

## Not Yet Implemented

| Component | Notes |
|---|---|
| Real PLN / URE inference | Requires a running OpenCog instance with Scheme rules loaded |
| Trained ML financial models (LSTM, etc.) | No datasets or training pipelines |
| Black-Scholes options pricing | Concept only |
| AML/KYC compliance monitoring | Not started |
| Real-time market data integration | Mock API calls only |
| Distributed AtomSpace (multi-node) | Single-instance only |
| Multi-currency support | USD only |

---

## Architecture Overview

```
src/                     ← Python integration package root
  core/                  ← AtomSpace bindings, GnuCash access, GGML kernels
  bridges/               ← Per-component ElizaOS ↔ OpenCog ↔ GnuCash bridges
  elizaos/               ← ElizaOS framework wrappers (actions, memory, models)
  financial/             ← Categorization, anomaly detection, NL query engine
  integration/           ← Master orchestration (HybridCognitiveFinancialFramework)
  microservices/         ← FastAPI endpoints, load balancer, Scheme grammar
  ml_pipeline/           ← ONNX, drift detection, sentiment analysis skeletons
  optimization/          ← Adaptive optimization, profiler stubs
  embodiment/            ← ROS / Unity3D interface stubs

libgnucash/              ← Upstream GnuCash C++ library (NOT a Python package)
gnucash/                 ← GnuCash GTK UI layer (C++)

tests/                   ← All test files (use pytest)
demos/                   ← All demo/example scripts

.github/workflows/
  python-ci.yml          ← Python unit tests CI (new)
  ci-tests.yml           ← GnuCash C++ CMake build CI (upstream)
```

---

## Running the Tests

```bash
# Install minimal Python dependencies
pip install numpy scikit-learn aiohttp pytest pytest-asyncio

# Run offline unit tests (no external services needed)
pytest tests/test_real_logic.py -v

# Run integration tests (requires live AtomSpace / ElizaOS servers)
pytest -m integration
```

---

## Next Development Priorities

1. **Connect real PLN server** — deploy `atomspace-restful` Docker image and point `ATOMSPACE_HOST`/`ATOMSPACE_PORT` env vars at it.
2. **Train financial ML models** — export historical transactions as CSV and train a proper LSTM or XGBoost forecasting model.
3. **Implement risk assessment logic** — replace placeholder `_assess_*` methods in `cognitive_analysis.py` with real calculations.
4. **Expand test coverage** — add edge-case tests for date-range queries, multi-account balance, and gzip XML parsing.
