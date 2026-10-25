# Project Audit Report — ML Weather Predictor and Dashboard

**Audit date:** 26 Oct 2026 · **Commit audited:** `c5e8b7c97eeb16e41ad0cc47f803ff1b4b378dab` (branch `main`, in sync
with `origin/main`, clean working tree before the documentation edits) · **Deployed site:**
<https://ml-weather-predictor-ae4j.onrender.com/>

**Scope.**
- **What was audited:** the source code, data, model artifacts, evaluation reports, notebook, frontend, tests,
  deployment configuration and the deployed service.
- **What was changed:** only `README.md`, `PROJECT_DEEP_DIVE.md` and this file. No source code, tests, models, data
  or dependencies were modified; nothing was committed or pushed.

**Evidence labels.**
- **Confirmed:** reproduced by running code, tests or requests during the audit.
- **Suspected:** plausible but not established.
- **Reported:** taken from the project owner's Render logs, which the auditor cannot access directly.

---

## 1. Executive summary

- **The ML pipeline is methodologically sound.** Features are leakage-free and shared between training and serving
  (tested). The test year was used only for final scoring, and every reported metric reproduces exactly from the saved
  models. The audit also recomputed the metrics independently, with targets and baselines built by date joins.
- **The model's skill is modest, and it is reported honestly.**
  - The ensemble beats climatology everywhere.
  - Against persistence it wins only in Delhi, and on RMSE in Bengaluru; it loses in Mumbai.
  - The 80% interval covers 72.3% (Bengaluru), 77.5% (Delhi) and 81.1% (Mumbai) on 2023.
- **The deployed service is healthy, but the live-data feature currently does not work there.** The weather provider
  answers the server's requests with HTTP 429 (daily limit exceeded), so every prediction on the public site is a
  clearly labelled demo snapshot from 25 Oct 2026. Live prediction works from a local machine.
- **No critical defects.** One high-severity operational issue (live data unavailable in production). Several
  medium- and low-severity issues: an incorrect EDA claim in the notebook, interval calibration, explainability
  scope, `/simulate` input handling, and stale comments.
- **The documentation was updated to match the code.** The README was stale on deployment status, Linux
  verification, test counts and one EDA claim. It has been rewritten, and two new documents were added.

## 2. Repository and deployment state

| Item | State | Evidence |
|---|---|---|
| Branch / commit | `main` at `c5e8b7c`, equal to `origin/main` | `git status -sb`, `git ls-remote` |
| History | 3 commits: initial commit, logging change, cache/rate-limit change | `git log` |
| Tracked files | 73 (72 in the initial commit + `tests/test_live_cache.py`), before this documentation work; no CI configuration (`.github/` absent) | `git ls-files` |
| Excluded locally | Course blueprint PDF, caches (`.gitignore`) | `git status --ignored` |
| Python / dependencies | 3.11.9; runtime pins in `requirements.txt` (8 packages); development pins in `requirements-dev.txt` | files |
| Deployment | Render free web service from `render.yaml`; `/health` → `healthy`, all 3 cities loaded, no artifact errors | Deployed `/health`, 26 Oct 2026 03:24 IST |
| Live data on deployment | **Not available**: `/predict` → `demo_snapshot`, `is_demo_fallback: true`, `fallback_reason: rate_limited` for all three cities, serving the snapshot captured 25 Oct 2026 01:07 IST (D = 24 Oct, target = 26 Oct). This is **not** a live prediction | Deployed `/predict`, checked several times between 03:15 and 03:30 IST, and again at **04:00 IST on 26 Oct 2026** |
| Latest public check (26 Oct 2026, 04:00 IST) | `/` → 200 after 43 s, which suggests the free instance was waking from sleep (not confirmed). `/health` → `healthy` (`timestamp_ist` 04:00:55); `/predict` → demo snapshot, `rate_limited`, for all cities. If the instance had restarted, its in-memory cooldown would have been empty, so `rate_limited` would reflect a fresh HTTP 429. That inference is **not verified** without the Render logs | `curl` from a local machine |
| Tests | 132 collected: **129 passed, 0 failed, 3 skipped** (opt-in live tests) | `python -m pytest tests/` |

## 3. Findings by severity

| ID | Severity | Finding | Status |
|---|---|---|---|
| H1 | **High** | Live weather data unavailable on the deployed site (provider HTTP 429, daily limit) | Confirmed (impact); cause **suspected** |
| M1 | Medium | Notebook states pre-monsoon is the hottest season in all three cities; false for Delhi | Confirmed |
| M2 | Medium | 80% interval under-covers in Bengaluru (72.3% / 69.6% unrounded) and varies 63–96% by season | Confirmed (known limitation) |
| M3 | Medium | Ensemble does not consistently beat persistence on the test year | Confirmed (known limitation) |
| M4 | Medium | Ensemble's validation MAE (the basis for choosing it) and p10/p90 reuse the same OOF predictions used to fit weights and choose hyperparameters | Confirmed design; size of optimism **not measured** |
| M5 | Medium | SHAP panel explains only LightGBM, which has 1.4% / 0.0% / 6.2% of the blend weight | Confirmed (labelled honestly in the UI) |
| L1 | Low | `/simulate` returns 500 for `nan`/`inf` inputs and accepts out-of-range values | Confirmed |
| L2 | Low | `/simulate` synthetic feature vector is internally inconsistent | Confirmed |
| L3 | Low | `/metrics` exposes three different coverage figures | Confirmed |
| L4 | Low | Demo `fallback_message` says "not a forecast for today" even when the snapshot's target date is today | Confirmed |
| L5 | Low | Stale or misleading code comments and naming | Confirmed |
| L6 | Low | Dashboard does not indicate that a response is `live_cached`; simulator omits Ridge | Confirmed |
| L7 | Low | Screenshots and demo snapshots are dated (old "(v3)" logo; snapshots from 25 Oct 2026, without `weather_data_fetched_at`) | Confirmed |
| L8 | Low | Responses with valid HTTP but incomplete data are not cached and start no backoff, so heavy traffic could repeat upstream calls | Suspected risk; not observed |
| L9 | Low | No CI; no frontend tests; retraining reproducibility untested | Confirmed |
| L10 | Low | EDA figure paths in `index.html` are absolute, unlike the residual figure | Confirmed; no effect on the current deployment |
| L11 | Low | Notebook "Team notes" section not yet written | Confirmed |

No **critical** findings: nothing produces wrong predictions silently, leaks test data, or exposes secrets.

## 4. Confirmed defects (detail)

**H1 — Live data unavailable in production.**
- **Evidence:**
  - Deployed `/predict` returns `data_source: "demo_snapshot"`, `fallback_reason: "rate_limited"` for Bengaluru,
    Delhi and Mumbai (26 Oct 2026).
  - Render logs (reported): `Open-Meteo returned HTTP 429 Too Many Requests … Daily API request limit exceeded. Please
    try again tomorrow.`
  - The same request returned 200 from a local machine, and the local live smoke tests passed.
- **Why it matters:** the headline feature, today's forecast, is not available on the public site. The app behaves
  correctly (an honest fallback), but a viewer sees only a saved snapshot.
- **Status:** the impact is confirmed. The **cause of the quota exhaustion is not established**: the known traffic was
  small, and a shared outgoing IP on the hosting platform is a hypothesis.
- **Action:**
  - After the provider reset, verify that `/predict` returns `live`, then `live_cached`.
  - If the limit is exhausted again despite low traffic, consider a dedicated outgoing IP, a provider API key or
    plan, or a scheduled once-daily fetch with persistent storage. Each of these needs approval.

**M1 — Incorrect EDA statement in the notebook.**
- **Evidence:**
  - `notebooks/weather_prediction_pipeline.ipynb`, markdown cell 13, and its generator `scripts/create_notebook.py`
    (line 96) state *"Pre-monsoon (March–May) is the hottest season in all three cities."* The generator's line 105
    (notebook cell 16) adds *"…in Delhi, where the hottest period (pre-monsoon) is dry…"*.
  - Recomputed from `data/delhi.csv`: Delhi's monsoon-season mean daily max is 34.6 °C, versus 34.3 °C in
    pre-monsoon. June (38.3 °C) is Delhi's hottest month.
  - Cell 16 also calls pre-monsoon Delhi's "hottest period".
- **Why it matters:** an examiner reading the notebook will find a factual error in the submitted analysis.
- **Action:** correct the wording in `scripts/create_notebook.py` and regenerate the notebook. This is outside the
  documentation files this task may edit; **needs approval**. The README no longer contains the claim, so the README
  and the notebook currently disagree. The README states the verified figures.

**L1 — `/simulate` input handling (Low).**
- **Location:** `backend/app.py` lines 394–400 (`simulate_scenario`). The parameters are plain `float = Query(…)`
  with no `ge` / `le` bounds and no finiteness check.
- **Evidence (local test client):**
  - `temp_yesterday=nan` → 500; `temp_yesterday=inf` → 500;
  - `humidity=250` → 200; `humidity=-50&temp_yesterday=60` → 200.
- **Why it matters:** unhandled server errors and physically meaningless inputs. The dashboard sliders limit normal
  users, but the API is public.
- **Action:** validate ranges with FastAPI `Query(ge=…, le=…)` and reject non-finite values with 422. **Needs
  approval** (code change).

**L2 — Inconsistent synthetic history in `/simulate` (Low).**
- **Location:** `backend/app.py` lines 422–425, which set `temp_roll_mean_7`, `temp_roll_std_7` and `temp_accel_1`
  by hand. The `predictions` block (line 454) returns `ensemble`, `lightgbm` and `random_forest`, but not Ridge.
- **Evidence:** with daily change m, the hand-built lags imply a 7-day mean offset of −2.17·m, but
  `temp_roll_mean_7` uses −1.5·m. The lags imply an acceleration of 0, but `temp_accel_1` uses 0.2·m. For m = 1, the
  lags' standard deviation is 1.27, but `temp_roll_std_7` uses 1.8.
- **Why it matters:** the models receive feature combinations that cannot occur in real data, which weakens the
  sensitivity analysis. The endpoint is labelled illustrative.
- **Action:** build a synthetic 7-day series and pass it through `make_features()`. **Needs approval.**

**L3 — Three coverage numbers in `/metrics` (Low).**
- **Location:** `src/train.py` lines 240–241 write `test_coverage_pct` and `quantile_coverage_pct` into
  `metrics.json`. `backend/app.py` line 476 adds `served_interval` from `evaluation_report.json`. `/metrics`
  returns all three.
- **Evidence:** the top-level `test_coverage_pct` is 69.6 / 77.3 / 79.7, from `src/train.py` with unrounded
  pre-rounding offsets. `served_interval.test_coverage_pct` is 72.3 / 77.5 / 81.1. `quantile_coverage_pct` is
  66.8 / 75.3 / 60.0.
- **Why it matters:** API consumers may quote the wrong figure. The dashboard uses the correct served figure, from
  `/predict` `interval_80.test_coverage_pct`.
- **Action:** document it (done in the deep dive), or rename or remove the legacy fields. **Needs approval.**

**L4 — Ambiguous fallback message (Low).**
- **Location:** `backend/app.py` line 369, in `demo_fallback()`.
- **Evidence:** on 26 Oct 2026 the deployed snapshot's `target_date` is 2026-10-26, while `fallback_message` says
  "It is not a forecast for today".
- **Why it matters:** it is technically a forecast *for* today made yesterday, so the wording can confuse. As the
  snapshot ages, the target date moves into the past and the problem disappears.
- **Action:** reword to "It is not a current forecast" and state the snapshot's target date. **Needs approval.**

**L5 — Stale or misleading comments and naming (Low).**
- **Evidence:**
  - The `src/train.py` docstring (line 11) and comment (line 180) present the quantile models as giving
    "weather-adaptive" intervals; the app does not serve them.
  - `src/train.py` line 30 imports `LinearRegression` but never uses it.
  - `src/features.py` line 122 calls the dew-point formula a "Magnus formula approximation"; it is the linear rule of
    thumb T − (100 − RH)/5, applied to daily max temperature and mean humidity.
  - The Ridge model is saved as `{city}_lr.joblib` and reported as `linear_regression`.
  - `src/ensemble.py` (line 3 and the class on line 10) names a "Stacking" ensemble; it is a weighted average, not a
    stacked meta-model.
  - `backend/app.py` line 163, `version="3.6.0"`, has no meaning.
- **Why it matters:** misleading during code review or a viva.
- **Action:** comment and docstring fixes are safe. Renaming files or keys would break the API and frontend contract
  and needs care. **Needs approval.**

**L6 — Presentation gaps.**
- **Evidence:** `frontend/app.js` never reads `data_source`, so `live_cached` looks the same as `live`. `/simulate`
  returns no Ridge prediction, although Ridge has 60–74% of the blend weight.
- **Action:** add a small "weather data fetched at …" label, and include Ridge in the simulator output. **Needs
  approval.**

## 5. Potential issues requiring more evidence

| Issue | What is known | What would settle it |
|---|---|---|
| Cause of the HTTP 429 daily-limit exhaustion | The known traffic was small; local requests succeeded at the same time | Observe after the provider reset whether the limit is exhausted again under negligible traffic; ask the provider; compare with a dedicated-IP deployment |
| Optimism of the ensemble's validation MAE (M4) | Weights (3 parameters) and the grid search (3 candidates) were fitted on the same 1210 OOF rows | Nested CV: fit weights on folds 1…k−1 and score fold k |
| Training vs. live data equivalence | 6 days spot-checked: max-temperature differences ≤ 0.3 °C, other variables identical | Compare a longer period (e.g. 60+ days) of archive vs. forecast-API past days |
| Bit-identical retraining | Seeds fixed (`random_state=42`); saved models reproduce metrics exactly | Retrain into a scratch directory and compare predictions |
| Repeated upstream calls on invalid data (L8) | Responses with valid HTTP but incomplete data are not cached and start no cooldown | Monitor logs for repeated `missing_values` / `insufficient_history` |
| Mobile layout on the deployed site | Verified locally at 390 px and 1440 px on 25 Oct 2026 | Re-check on a phone against the public URL |

## 6. Correctness and consistency of the ML pipeline

| Check | Result | Evidence |
|---|---|---|
| Target definition | T_max on D+2; `FORECAST_HORIZON_DAYS = 2`; shared constant | `src/features.py`; `test_target_is_max_temperature_two_days_after_d` |
| Same features in training and serving | Yes, `make_features()` in both; the served prediction equals the saved ensemble on the training-time row | `test_serving_matches_training_features_and_models` |
| Gaps and duplicates | Rejected (`ValueError` offline; `WeatherDataError` live) | `test_non_consecutive_dates_are_rejected`, `test_missing_*` |
| IST date logic on a UTC server | Correct around midnight IST | `test_utc_server_*`; deployed `timestamp_ist` |
| Saved metrics reproducible | Max difference 0.0 between `metrics.json` and recomputation | `evaluation_report.json`; `test_saved_metrics_are_reproducible_from_artifacts`; independent recomputation during the audit |
| Interval served = interval evaluated | Same `interval_bounds()` function | `src/intervals.py`; `test_saved_evaluation_report_matches_recomputation` |
| Model artifacts compatible | All 18 joblib files load with scikit-learn 1.9.1 / LightGBM 4.7.0; 22 features each | `test_all_model_artifacts_load_with_expected_feature_count`; deployed `/health` |
| Collinearity | Rank 19 of 22; `temp_momentum_1`, `temp_accel_1`, `temp_roll_mean_7` are linear combinations of the lags; `dew_point_lag_0` is affine | Numerical rank check |
| Ridge vs. plain linear regression | Ridge validation MAE lower in all cities (0.990 vs 1.012; 1.530 vs 1.581; 1.129 vs 1.157); scaling changes Ridge by ≤ 0.020 °C | Re-run during the audit |

## 7. Data leakage and evaluation validity

- **No test-year leakage found.** The climatology comes from training years only (tested). Features use only
  D−6 … D (tested by truncation). The split is by target date. Hyperparameters, weights and interval offsets come from
  training-year folds.
- **Fold boundaries are legitimate.** The values T(t+1), T(t+2) that are training targets near a fold boundary later
  appear only as *features* of validation rows, i.e. information available at forecast time.
- **Validity caveats:**
  - M4: OOF reuse.
  - The EDA, including its 7-day-window diagnostics, used all five years, including 2023.
  - One test year of autocorrelated days, with no significance testing.
  - A distribution shift in 2023: warmer by 0.90 °C (Bengaluru) and 1.12 °C (Mumbai) than 2019–2022.

## 8. Model results versus baselines (2023 test year)

| City | Ensemble MAE / RMSE | Persistence | Climatology | Beats persistence? | Beats climatology? | Lowest test MAE |
|---|---|---|---|---|---|---|
| Bengaluru | 1.024 / 1.280 | 1.010 / 1.376 | 1.737 / 2.177 | RMSE only | Yes | Persistence |
| Delhi | 1.502 / 1.994 | 1.616 / 2.188 | 2.616 / 3.287 | Yes (both) | Yes | Ensemble |
| Mumbai | 1.215 / 1.504 | 1.068 / 1.480 | 1.920 / 2.352 | No | Yes | Persistence |

Validation MAE (training years) ranked the ensemble first in every city: 0.981 / 1.504 / 1.095 versus persistence
1.115 / 1.588 / 1.200. Ridge alone was within 0.01–0.03 °C of the ensemble in validation and slightly better on the
test year in Bengaluru and Mumbai. All values come from `models/metrics.json` and `models/evaluation_report.json`, and
were independently recomputed in the audit.

## 9. Backend and API correctness

| Endpoint | Verified behaviour (local tests and/or deployed) |
|---|---|
| `/` | 200 dashboard (deployed) |
| `/health` | 200 `healthy` with loaded cities (deployed); 503 when artifacts are missing (test) |
| `/cities` | 200 `["Bengaluru","Delhi","Mumbai"]` (deployed) |
| `/metrics` | 200 (deployed); 400 for an unknown city (test) |
| `/predict` | Deployed: 200 demo snapshot (`rate_limited`); 400 for `Paris`. Local: `live` then `live_cached` with the same `weather_data_fetched_at` (real API, 26 Oct 2026). Tests: 503 for missing models or no snapshot; 500 for internal errors |
| `/simulate` | 200 (deployed); 400 for an unknown city (test); **500 for non-finite inputs (L1)** |
| `/docs`, `/openapi.json` | 200 (deployed) |

Error semantics are deliberate and tested: only a data or provider problem (`WeatherDataError`) leads to the demo
snapshot, and the snapshot is never re-dated or mutated.

## 10. Live weather integration and fallback behaviour

- **Validation:** exactly D−6 … D must be present and numeric; today's partial day and the forecast rows are never
  inputs. Confirmed by tests.
- **Cache:**
  - **Key** `(city, D, 10, 4)`; **TTL** 30 minutes on a monotonic clock.
  - **Only validated payloads are stored**, and they are re-validated on each hit; callers get deep copies.
  - **One request per city:** concurrent requests share one upstream call. Confirmed: 8 concurrent → 1 call.
- **HTTP 429 cooldown:**
  - **Length:** `Retry-After` (seconds or HTTP date), or 15 minutes, capped at 24 hours.
  - **Scope:** all cities. Confirmed: 9 requests during the cooldown → 0 upstream calls.
  - **Edge case:** up to 3 simultaneous requests (one per city) can occur before the first 429 arrives.
- **Restart and sleep:** the cache and the cooldown are in memory and reset on restart or wake-up. Not a defect, but
  a reliability limit.
- **The cache does not bypass the quota.** It only reduces call volume; with an exhausted quota, users get the
  labelled snapshot. Confirmed on the deployed site.

## 11. Frontend and UX correctness

- **Rendering is correct:**
  - live fields render, missing numbers show as "--", and all API text is escaped;
  - responses for a city the user has already left are ignored;
  - the demo banner is driven by `is_demo_fallback`;
  - the error panel clears stale values;
  - "better/worse than persistence" outcomes are computed from `/metrics`, not hard-coded.
- **Labels are honest:** SHAP is marked "LightGBM component" with its weight; the simulator is marked illustrative;
  the official forecast is marked as a reference.
- **Gaps:** L6 (cached data not marked; Ridge missing from the simulator) and L10 (absolute EDA paths).
- **Accessibility:** ARIA tabs, keyboard support for the tabs and the lightbox, focus styles and reduced-motion
  support are present. Not audited with assistive technology.

## 12. Tests and coverage gaps

- **Suite:** 132 tests in 6 modules.
  - `python -m pytest tests/` → **129 passed, 0 failed, 3 skipped** (26 Oct 2026). One harmless Starlette deprecation
    warning.
  - `RUN_LIVE_TESTS=1 python -m pytest tests/ -k live_open_meteo` → 3 passed (local machine, 26 Oct 2026).
- **Strengths:** leakage, date logic, validation, real-model execution, fallback semantics, cache, cooldown and
  concurrency, and reproduction of the saved metrics.
- **Gaps:**
  - no CI;
  - no automated frontend tests;
  - no tests for `/simulate` with invalid inputs (L1);
  - no retraining reproducibility test;
  - no tests of `src/eda.py` or `streamlit_app.py`;
  - provider behaviour beyond the opt-in smoke tests is mocked.
- **Flakiness risk:** the concurrency test depends on a 0.2 s sleep. It passed 20/20 runs.
- **One unexplained run:** during development, one run reported 354 warnings, which did not recur in 8 later runs.

## 13. Security and repository hygiene

| Check | Result |
|---|---|
| Secrets, API keys, tokens in tracked files | None found (pattern search). No keys are needed: Open-Meteo is used without authentication |
| Personal data and local absolute paths | None in tracked files, including notebook outputs (pattern search for user names, drive paths and e-mail addresses) |
| Commit identity | GitHub no-reply address |
| `.gitignore` | Excludes caches, virtual environments, `.env*`, key files, logs, the course blueprint PDF |
| CORS | Off by default (same-origin); optional `ALLOWED_ORIGINS`, GET only, no credentials |
| Error disclosure | 500 responses are generic; provider error details go only to logs, without URL or query string, and with bodies capped at 200 characters |
| Pickle/joblib loading | Loads only repository artifacts; replacing them is equivalent to a code change |
| Abuse resistance | No authentication or rate limiting on the app's own endpoints; the cache limits upstream calls while data are valid (see L8 for invalid data) |
| Repository size | About 42 MB; largest files about 13 MB (`delhi_rf.joblib`, `delhi_ensemble.joblib`) |

## 14. Documentation discrepancies corrected

| Previous README statement | Correction | Evidence |
|---|---|---|
| "Deployment status: pending … There is no public URL yet." | Deployed at the public URL; healthy; live data currently rate-limited | Deployed `/health`, `/predict` |
| "Loading the models … on Render's Linux servers has not been tested yet." | Models load on Render (`healthy`, no artifact errors) | Deployed `/health` |
| "Last local run … 99 passed, 3 skipped" | 132 collected, 129 passed, 3 skipped; `test_live_cache.py` added to the test table | pytest run, 26 Oct 2026 |
| "Pre-monsoon (Mar–May) is the hottest season everywhere" | True for Bengaluru and Mumbai; in Delhi, monsoon (34.6 °C) ≈ pre-monsoon (34.3 °C), and June is the hottest month | Recomputed from `data/delhi.csv` |
| "Three features (daily change, its change and the approximate dew point) are exact linear combinations" | The three exact linear dependencies are the daily change, its change and the 7-day mean; dew point is affine (redundant with an intercept) | Numerical rank check |
| Objectives section referenced the blueprint PDF by file name | Says the blueprint is not included | — |
| No description of current live-data status, `live_cached`, or the incident | Added, with evidence and without asserting the unproven cause | Sections 13 and 15 of the README |

## 15. Recommended fixes, prioritised by impact

1. **Restore and monitor live data (H1).** After the provider reset, check `/predict` and the logs. If exhaustion
   recurs under low traffic, choose among:
   - a scheduled daily fetch with a persistent store;
   - a provider API key or plan;
   - a host with a dedicated outgoing IP.
2. **Correct the notebook's season claim (M1)** in `scripts/create_notebook.py` and regenerate the notebook.
3. **Harden `/simulate` (L1, L2):** range and finiteness validation; build its features through `make_features()`
   from a synthetic series; include the Ridge output.
4. **Clarify the API and UI (L3, L4, L6):**
   - remove or rename the legacy coverage fields in `/metrics`;
   - reword the fallback message;
   - show "weather data fetched at …" for cached responses.
5. **Clean up comments (L5):** the train.py docstring, the dew-point comment, the unused import.
6. **Methodology (M2–M5), as future work:**
   - nested CV to quantify the ensemble's optimism;
   - interval calibration chosen on validation data (e.g. conformal), then a new test period;
   - more training years and evaluation over several years;
   - explaining the full ensemble.
7. **Engineering (L9):** CI that runs `pytest` on push; a retraining-reproducibility check; refreshed screenshots and
   snapshots (L7).

## 16. Issues that need approval before code changes

All items below change source code, tests, generated notebooks or API behaviour. **None were changed in this audit.**

| Item | Files likely affected | Behavioural impact |
|---|---|---|
| M1 notebook wording | `scripts/create_notebook.py`, `notebooks/weather_prediction_pipeline.ipynb` (regenerated) | Text only; the notebook is re-executed |
| L1/L2 `/simulate` validation and consistent features | `backend/app.py`, `tests/test_api.py` | Invalid inputs → 422; simulator outputs change |
| L3 coverage fields | `backend/app.py` (and possibly `src/train.py` output) | `/metrics` schema change |
| L4 fallback message | `backend/app.py`, tests | Text change |
| L5 comments and unused import | `src/train.py`, `src/features.py` | None |
| L6 cached-data label; Ridge in simulator | `frontend/app.js`, `frontend/index.html`, `backend/app.py` | UI and API additions |
| H1 provider and infrastructure changes | Deployment, possibly `backend/live_data.py` | Cost or hosting decisions |
| Any retraining | `models/*`, `figures/residuals_*`, `metrics.json`, `evaluation_report.json` | All results change; needs re-evaluation and documentation |

## 17. Final readiness assessment

Status scale: **Ready** (no action needed for the stated purpose) · **Ready with caveats** (usable; limitations must
be stated) · **Not ready** (blocks the stated purpose).

| Area | Status | Explanation |
|---|---|---|
| Code correctness | **Ready with caveats** | The prediction path is tested end to end with real models; no silent-failure paths found. Caveats: `/simulate` input handling (L1/L2), minor stale comments |
| ML methodology | **Ready with caveats** | Chronological split, leakage controls, baselines and OOF-based selection are sound. Caveats: OOF reuse (M4), a small grid, untuned Ridge α, a fixed-width interval |
| Evaluation credibility | **Ready with caveats** | Every number reproduces from the artifacts and was independently recomputed; weaknesses are reported. Caveats: one test year, no significance tests, coverage below target in Bengaluru |
| Test coverage | **Ready with caveats** | 129 passing offline tests covering the critical paths. Gaps: no CI, no frontend tests, mocked provider |
| Local reproducibility | **Ready** | Pinned dependencies and committed data and models; the app, tests and live prediction ran locally on 26 Oct 2026. Retraining was not re-run (it would overwrite artifacts) |
| Deployment reliability | **Ready with caveats** | The Render service builds, loads the models and serves all endpoints. Free-tier sleep; in-memory cache |
| Live data availability | **Not ready** | The provider currently rejects the server's requests (HTTP 429, daily limit). The site shows only the labelled demo snapshot until this is resolved |
| Documentation quality | **Ready** | The README, deep dive and this report agree with the code and the verified results; facts and assumptions are labelled. The notebook still needs the M1 correction and the team's own notes |
| Viva readiness | **Ready with caveats** | Strong, honest story with a full question bank and a demo script robust to the fallback. Fix M1 before submission, and prepare to explain H1, M2, M3 and M5 openly |
