/**
 * frontend/app.js
 * Dashboard logic:
 * - API health check (also wakes a sleeping free-tier server)
 * - Live prediction, 80% interval, baselines, current conditions, official forecast
 * - Test-year comparison table with outcomes computed from /metrics (no hard-coded claims)
 * - Model inputs (the 7 completed days used), TreeSHAP of the LightGBM component
 * - Illustrative what-if simulator
 * - EDA figures, accessible tabs and image lightbox
 */

// Same-origin by default (FastAPI serves this page). A separately hosted page can set
// window.APP_CONFIG = { apiBaseUrl: "https://..." } before loading this script.
const API_BASE_URL = (window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl) || window.location.origin;
let currentCity = "Bengaluru";
let simDebounceTimer = null;
let latestRequestId = 0;

const MODEL_ROWS = [
    { key: "ensemble", prefix: "ens" },
    { key: "linear_regression", prefix: "lr" },
    { key: "random_forest", prefix: "rf" },
    { key: "lightgbm", prefix: "lgb" },
];

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initCitySelector();
    initImageLightbox();
    initSimulator();
    document.getElementById("retry-btn").addEventListener("click", () => loadCityData(currentCity));
    checkApiHealth();
    loadCityData(currentCity);
});

// --- Helpers ---
function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function fmtNum(value, digits = 1, unit = "") {
    return (value === null || value === undefined || Number.isNaN(value)) ? "unavailable" : `${Number(value).toFixed(digits)}${unit}`;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// --- Tabs (ARIA tab pattern: click, Left/Right/Home/End keys) ---
function initTabs() {
    const tabBtns = Array.from(document.querySelectorAll(".tab-btn"));

    function activate(btn, focus) {
        tabBtns.forEach(b => {
            const selected = b === btn;
            b.classList.toggle("active", selected);
            b.setAttribute("aria-selected", String(selected));
            b.tabIndex = selected ? 0 : -1;
            const pane = document.getElementById(b.dataset.tab);
            if (pane) pane.classList.toggle("active", selected);
        });
        if (focus) btn.focus();
    }

    tabBtns.forEach((btn, idx) => {
        btn.addEventListener("click", (e) => { e.preventDefault(); activate(btn, false); });
        btn.addEventListener("keydown", (e) => {
            const moves = { ArrowRight: idx + 1, ArrowLeft: idx - 1, Home: 0, End: tabBtns.length - 1 };
            if (e.key in moves) {
                e.preventDefault();
                activate(tabBtns[(moves[e.key] + tabBtns.length) % tabBtns.length], true);
            }
        });
    });
}

// --- City Selector ---
function initCitySelector() {
    document.getElementById("city-select").addEventListener("change", (e) => {
        currentCity = e.target.value;
        loadCityData(currentCity);
    });
}

// --- API Health Check ---
async function checkApiHealth() {
    const indicator = document.querySelector(".status-indicator");
    try {
        const resp = await fetch(`${API_BASE_URL}/health`);
        const healthy = resp.ok;
        setText("status-text", healthy ? "Models loaded" : "Server degraded");
        indicator.classList.toggle("offline", !healthy);
    } catch (err) {
        setText("status-text", "Server unreachable");
        indicator.classList.add("offline");
    }
}

// --- Load City Data ---
async function loadCityData(city) {
    const requestId = ++latestRequestId;  // ignore responses for a city the user already left
    const loadingState = document.getElementById("loading-state");
    const errorPanel = document.getElementById("error-state");
    loadingState.classList.remove("hidden");
    errorPanel.classList.add("hidden");

    const residualImg = document.getElementById("city-residual-img");
    if (residualImg) residualImg.src = `${API_BASE_URL}/figures/residuals_${city.toLowerCase()}.png`;

    try {
        const [predictResp, metricsResp] = await Promise.all([
            fetch(`${API_BASE_URL}/predict?city=${encodeURIComponent(city)}`),
            fetch(`${API_BASE_URL}/metrics?city=${encodeURIComponent(city)}`)
        ]);
        if (requestId !== latestRequestId) return;

        if (!predictResp.ok) {
            let detail = `server returned status ${predictResp.status}`;
            try {
                const body = await predictResp.json();
                if (typeof body.detail === "string") detail = body.detail;
            } catch (e) { /* non-JSON error body */ }
            throw new Error(detail);
        }

        const predictData = await predictResp.json();
        const metricsData = metricsResp.ok ? await metricsResp.json() : null;
        if (requestId !== latestRequestId) return;

        renderDemoBanner(predictData);
        renderPrediction(predictData, metricsData);
        renderInputWindow(predictData.input_window || []);
        renderShapAttribution(predictData.explainable_ai, metricsData);
        if (metricsData) {
            renderMetrics(predictData, metricsData);
        } else {
            setText("benchmark-summary", "Test-year metrics could not be loaded.");
        }
        syncSimulatorWithCity(predictData);
    } catch (error) {
        if (requestId !== latestRequestId) return;
        console.error("Error loading city data:", error);
        showError(`${city}: ${error.message.replace(/\.$/, "")}. No live or demo values are shown.`);
    } finally {
        if (requestId === latestRequestId) loadingState.classList.add("hidden");
    }
}

function showError(message) {
    document.getElementById("demo-banner").classList.add("hidden");
    setText("error-text", message);
    document.getElementById("error-state").classList.remove("hidden");
    // Clear values from a previous city so nothing stale is presented as current
    ["predicted-temp-val", "official-temp-val"].forEach(id => setText(id, "--.-"));
    setText("interval-range-text", "[--.-°C, --.-°C]");
    setText("target-date-display", "Target date: --");
    setText("completed-date-display", "Last completed day D: --");
    ["cond-temp", "cond-humidity", "cond-wind", "cond-dewpoint"].forEach(id => setText(id, "--"));
    document.getElementById("input-window-body").innerHTML = `<tr><td colspan="5">--</td></tr>`;
}

function renderDemoBanner(data) {
    const banner = document.getElementById("demo-banner");
    if (data.is_demo_fallback) {
        document.getElementById("banner-text").innerHTML =
            `<strong>Demo snapshot, not today's forecast.</strong> ${escapeHtml(data.fallback_message || "")}`;
        banner.classList.remove("hidden");
    } else if (data.data_source === "live_stale") {
        // Real same-day observations, reused because the weather service is rate-limiting requests
        const fetchedAt = data.weather_data_fetched_at
            ? `${data.weather_data_fetched_at.replace("T", " ").slice(0, 16)} IST` : "an earlier time today";
        const reason = data.stale_reason === "rate_limited"
            ? "the weather service is limiting requests" : "fresh weather data could not be fetched";
        document.getElementById("banner-text").innerHTML =
            `<strong>Using earlier live data.</strong> Because ${escapeHtml(reason)}, this forecast uses ` +
            `weather data fetched at ${escapeHtml(fetchedAt)}. The current conditions and the official forecast ` +
            `shown below also come from that earlier fetch and may be out of date.`;
        banner.classList.remove("hidden");
    } else {
        banner.classList.add("hidden");
    }
}

// --- Main prediction card, current conditions, official reference ---
function renderPrediction(data, metricsData) {
    setText("hero-city-name", data.city);
    setText("target-date-display", `Target date: ${data.target_date}${data.is_demo_fallback ? " (snapshot)" : " (tomorrow, India)"}`);
    setText("completed-date-display", `Last completed day D: ${data.last_completed_date}`);
    setText("predicted-temp-val", data.predicted_temp.toFixed(1));

    const interval = data.interval_80;
    setText("interval-range-text", `[${interval.lower.toFixed(1)}°C, ${interval.upper.toFixed(1)}°C]`);
    const coverage = interval.test_coverage_pct;
    setText("interval-city-stats",
        `Error offsets [${interval.p10_offset.toFixed(2)}°C, +${interval.p90_offset.toFixed(2)}°C] • ` +
        (coverage === null || coverage === undefined
            ? "coverage not evaluated"
            : `actual value fell inside on ${coverage.toFixed(1)}% of ${interval.test_days} test days in 2023 (target 80%)`));

    const cc = data.current_conditions || {};
    setText("cond-temp", fmtNum(cc.temperature, 1, " °C"));
    setText("cond-humidity", fmtNum(cc.humidity, 0, " %"));
    setText("cond-wind", fmtNum(cc.wind_speed, 1, " km/h"));
    setText("cond-dewpoint", fmtNum(cc.dew_point, 1, " °C"));
    setText("conditions-note",
        `Readings${cc.observed_at ? ` at ${cc.observed_at} (IST)` : ""} are for information only. The model uses only ` +
        `fully completed days up to D (yesterday in India); today's partial readings are never model inputs.`);

    const official = data.official_forecast || {};
    const hasOfficial = official.temperature_max !== null && official.temperature_max !== undefined;
    setText("official-temp-val", hasOfficial ? official.temperature_max.toFixed(1) : "Not available");
    document.getElementById("official-unit").classList.toggle("hidden", !hasOfficial);
    setText("official-label", `Open-Meteo numerical weather forecast for ${official.date || data.target_date}`);
}

// --- Real observations used as model inputs ---
function renderInputWindow(rows) {
    const body = document.getElementById("input-window-body");
    if (!rows.length) {
        body.innerHTML = `<tr><td colspan="5">Not available</td></tr>`;
        return;
    }
    body.innerHTML = rows.map(r => `
        <tr>
            <td>${escapeHtml(r.date)}</td>
            <td>${r.temperature_2m_max.toFixed(1)}</td>
            <td>${r.relative_humidity_2m_mean.toFixed(0)}</td>
            <td>${r.wind_speed_10m_max.toFixed(1)}</td>
            <td>${r.precipitation_sum.toFixed(1)}</td>
        </tr>`).join("");
}

// --- TreeSHAP attributions of the LightGBM component ---
function renderShapAttribution(shapData, metricsData) {
    const container = document.getElementById("shap-drivers-container");
    container.innerHTML = "";
    const weight = metricsData && metricsData.ensemble_weights ? metricsData.ensemble_weights[0] : null;
    setText("shap-weight-note", weight === null
        ? "LightGBM component"
        : `LightGBM carries ${(weight * 100).toFixed(1)}% of the ensemble weight for this city`);

    const drivers = (shapData && shapData.top_drivers) || [];
    if (drivers.length === 0) {
        container.innerHTML = `<p class="shap-desc">No feature contributions available.</p>`;
        return;
    }
    const maxDelta = Math.max(...drivers.map(d => Math.abs(d.contribution)), 0.5);
    drivers.forEach(driver => {
        const isWarming = driver.contribution >= 0;
        const item = document.createElement("div");
        item.className = "shap-item";
        item.innerHTML = `
            <div class="shap-meta">
                <div>
                    <span class="shap-label">${escapeHtml(driver.label)}</span>
                    <span class="shap-value-badge">• Value: ${escapeHtml(driver.value)}</span>
                </div>
                <span class="shap-delta ${isWarming ? "warming" : "cooling"}">${isWarming ? "+" : ""}${driver.contribution.toFixed(2)} °C (${driver.direction})</span>
            </div>
            <div class="shap-bar-wrapper">
                <div class="shap-bar ${isWarming ? "warming" : "cooling"}" style="width: ${Math.round(Math.abs(driver.contribution) / maxDelta * 100)}%;"></div>
            </div>
            <div class="shap-desc">${escapeHtml(driver.explanation)}</div>`;
        container.appendChild(item);
    });
}

// --- Comparison table and seasonal breakdown (all outcomes computed from /metrics) ---
function outcomePill(modelMae, persistenceMae) {
    const diff = modelMae - persistenceMae;
    if (Math.abs(diff) < 0.0005) return `<span class="status-pill simple">Tie</span>`;
    return diff < 0
        ? `<span class="status-pill win">Better by ${(-diff).toFixed(2)} °C</span>`
        : `<span class="status-pill loss">Worse by ${diff.toFixed(2)} °C</span>`;
}

const MODEL_NAMES = {
    ensemble: "the ensemble", linear_regression: "Ridge regression", random_forest: "the Random Forest",
    lightgbm: "LightGBM", persistence: "the persistence baseline", climatology: "the climatology baseline",
};

function renderMetrics(predictData, metricsData) {
    const overall = metricsData.overall;
    const breakdown = predictData.model_breakdown || {};

    MODEL_ROWS.forEach(({ key, prefix }) => {
        setText(`table-${prefix}-pred`, fmtNum(breakdown[key], 1, " °C"));
        setText(`table-${prefix}-mae`, `${overall[key].mae.toFixed(2)} °C`);
        setText(`table-${prefix}-rmse`, `${overall[key].rmse.toFixed(2)} °C`);
        document.getElementById(`table-${prefix}-outcome`).innerHTML = outcomePill(overall[key].mae, overall.persistence.mae);
    });
    setText("table-pers-pred", `${predictData.persistence_baseline.toFixed(1)} °C`);
    setText("table-clim-pred", `${predictData.climatology_baseline.toFixed(1)} °C`);
    ["persistence", "climatology"].forEach(key => {
        const prefix = key === "persistence" ? "pers" : "clim";
        setText(`table-${prefix}-mae`, `${overall[key].mae.toFixed(2)} °C`);
        setText(`table-${prefix}-rmse`, `${overall[key].rmse.toFixed(2)} °C`);
    });

    // One honest sentence summarising the comparison
    const beats = metricsData.main_model_beats;
    let summary = "";
    if (beats) {
        const vsPers = beats.persistence.mae && beats.persistence.rmse ? "beats persistence on both MAE and RMSE"
            : beats.persistence.mae ? "beats persistence on MAE but not RMSE"
            : beats.persistence.rmse ? "beats persistence on RMSE but not on MAE"
            : "does not beat persistence on either MAE or RMSE";
        summary = `In ${predictData.city}, the ensemble ${vsPers}` +
            `${beats.climatology.mae && beats.climatology.rmse ? ", and clearly beats climatology" : ""}. `;
    }
    if (metricsData.best_model_by_test_mae) {
        summary += `Lowest test MAE: ${MODEL_NAMES[metricsData.best_model_by_test_mae] || metricsData.best_model_by_test_mae}. `;
    }
    if (metricsData.validation && metricsData.validation.mae) {
        const v = metricsData.validation.mae;
        summary += `The ensemble was chosen on validation folds from the training years (validation MAE ${v.ensemble.toFixed(2)} °C vs persistence ${v.persistence.toFixed(2)} °C), not on 2023.`;
    }
    setText("benchmark-summary", summary);

    const seasonsContainer = document.getElementById("seasons-metrics-grid");
    seasonsContainer.innerHTML = "";
    for (const [season, s] of Object.entries(metricsData.by_season || {})) {
        const card = document.createElement("div");
        card.className = "season-card";
        card.innerHTML = `
            <div class="season-name">
                <span>${escapeHtml(season)}</span>
                <span class="season-sub">${s.count} test days</span>
            </div>
            <div class="season-mae-val">${s.ensemble.mae.toFixed(2)} °C <span class="season-mae-label">ensemble MAE</span></div>
            <div class="season-compare">
                Persistence ${s.persistence.mae.toFixed(2)} °C • Climatology ${s.climatology.mae.toFixed(2)} °C • Ridge ${s.linear_regression.mae.toFixed(2)} °C
            </div>`;
        seasonsContainer.appendChild(card);
    }
}

// --- Illustrative what-if simulator ---
function initSimulator() {
    ["sim-temp-slider", "sim-rh-slider", "sim-wind-slider", "sim-momentum-slider"].forEach(id => {
        document.getElementById(id).addEventListener("input", () => {
            updateSimulatorLabels();
            triggerSimulationDebounced();
        });
    });
}

function updateSimulatorLabels() {
    const temp = parseFloat(document.getElementById("sim-temp-slider").value);
    const rh = parseFloat(document.getElementById("sim-rh-slider").value);
    const wind = parseFloat(document.getElementById("sim-wind-slider").value);
    const mom = parseFloat(document.getElementById("sim-momentum-slider").value);
    setText("sim-temp-val", `${temp.toFixed(1)} °C`);
    setText("sim-rh-val", `${rh} %`);
    setText("sim-wind-val", `${wind.toFixed(1)} km/h`);
    setText("sim-momentum-val", `${mom >= 0 ? "+" : ""}${mom.toFixed(1)} °C/day`);
}

function syncSimulatorWithCity(predictData) {
    // Start the sliders from the real day-D observation used by the model
    const inputs = predictData.input_window || [];
    const dayD = inputs[inputs.length - 1];
    if (!dayD) return;
    document.getElementById("sim-temp-slider").value = dayD.temperature_2m_max;
    document.getElementById("sim-rh-slider").value = dayD.relative_humidity_2m_mean;
    document.getElementById("sim-wind-slider").value = dayD.wind_speed_10m_max;
    updateSimulatorLabels();
    triggerSimulationDebounced();
}

function triggerSimulationDebounced() {
    clearTimeout(simDebounceTimer);
    simDebounceTimer = setTimeout(runSimulation, 180);
}

async function runSimulation() {
    const params = new URLSearchParams({
        city: currentCity,
        temp_yesterday: document.getElementById("sim-temp-slider").value,
        humidity: document.getElementById("sim-rh-slider").value,
        wind_speed: document.getElementById("sim-wind-slider").value,
        temp_momentum: document.getElementById("sim-momentum-slider").value,
    });
    try {
        const resp = await fetch(`${API_BASE_URL}/simulate?${params}`);
        if (!resp.ok) throw new Error(`status ${resp.status}`);
        const data = await resp.json();
        setText("sim-result-temp", data.predictions.ensemble.toFixed(1));
        setText("sim-result-dewpoint", `${data.simulated_inputs.calculated_dew_point.toFixed(1)} °C`);
        setText("sim-result-lgb", `${data.predictions.lightgbm.toFixed(1)} °C`);
        setText("sim-result-rf", `${data.predictions.random_forest.toFixed(1)} °C`);
        const top = data.explainable_ai && data.explainable_ai.top_drivers[0];
        document.getElementById("sim-top-driver-text").innerHTML = top
            ? `<strong>${escapeHtml(top.label)}</strong> (value ${escapeHtml(top.value)}) changes the LightGBM output by ` +
              `<strong>${top.contribution >= 0 ? "+" : ""}${top.contribution.toFixed(2)} °C</strong>.`
            : "No feature contributions available.";
    } catch (err) {
        console.error("Simulation error:", err);
        setText("sim-result-temp", "--.-");
        setText("sim-top-driver-text", "The simulator could not reach the server.");
    }
}

// --- Image lightbox for figures (mouse and keyboard) ---
function initImageLightbox() {
    const modal = document.getElementById("image-modal");
    const modalImg = document.getElementById("modal-img");
    const closeBtn = document.getElementById("modal-close");
    let lastFocused = null;

    function openModal(wrapper) {
        const img = wrapper.querySelector("img");
        const card = wrapper.closest(".eda-card");
        if (!img) return;
        lastFocused = wrapper;
        modalImg.src = img.src;
        modalImg.alt = img.alt;
        setText("modal-title", card ? card.querySelector("h3").textContent : "Figure");
        setText("modal-desc", card ? card.querySelector(".eda-desc").textContent : "");
        modal.classList.remove("hidden");
        document.body.style.overflow = "hidden";
        closeBtn.focus();
    }

    function closeModal() {
        modal.classList.add("hidden");
        document.body.style.overflow = "";
        if (lastFocused) lastFocused.focus();
    }

    document.querySelectorAll(".eda-card .img-wrapper").forEach(wrapper => {
        wrapper.addEventListener("click", () => openModal(wrapper));
        wrapper.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModal(wrapper); }
        });
    });
    closeBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !modal.classList.contains("hidden")) closeModal();
    });
}
