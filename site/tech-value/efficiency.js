// Performance per watt page: CPU/GPU switch, frontier, ranking with an EDP weight, cost of
// ownership calculator and launch-year trends, for site/tech-value/efficiency.qmd.
//
// The page embeds the data as JSON (built by pdi/tech_value/efficiency.py). The calculations at
// the top mirror the Python ones and are checked against them by tests/test_efficiency.py,
// which runs this file under Node.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- calculations
  function secondsPerJob(score, samples) {
    return (samples / score) * 60;
  }
  function energyPerJob(score, watts, samples) {
    return watts * secondsPerJob(score, samples);
  }
  // Energy-delay product with a weight on time: alpha 1 is EDP, alpha 2 is ED²P.
  function edp(score, watts, samples, alpha) {
    return energyPerJob(score, watts, samples) * Math.pow(secondsPerJob(score, samples), alpha);
  }
  // Purchase price plus electricity for the hours under load.
  function tco(price, watts, usdPerKwh, hoursPerDay, loadShare, years) {
    const kwh = (watts * hoursPerDay * loadShare * 365 * years) / 1000;
    const power = kwh * usdPerKwh;
    return { total: price + power, power: power };
  }
  function frontier(rows) {
    return rows.map(
      (r) =>
        !rows.some(
          (o) =>
            o !== r &&
            o.score >= r.score &&
            o.watts <= r.watts &&
            (o.score > r.score || o.watts < r.watts)
        )
    );
  }

  const calc = { secondsPerJob, energyPerJob, edp, tco, frontier };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = calc;
    return;
  }
  root.EFcalc = calc;

  // ---------------------------------------------------------------- page
  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtUsd = (x) => "$" + Math.round(x).toLocaleString("en-US");
  const fmt1 = (x) => x.toLocaleString("en-US", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const fmtE = (x) => x.toLocaleString("en-US", { maximumFractionDigits: x < 10 ? 2 : 1, minimumFractionDigits: x < 10 ? 2 : 1 });
  const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
  const fmtPct = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(0) + "%";
  const VENDOR = { amd: "AMD", intel: "Intel", nvidia: "NVIDIA" };
  const SLOT = { intel: 1, amd: 2, nvidia: 3 };
  const CONFIG = { displayModeBar: false, responsive: true };
  const W_TICKS = [10, 20, 35, 50, 75, 100, 150, 200, 300, 450, 600];
  const S_TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

  function boot() {
    const el = $("ef-data");
    if (!el) return;
    const D = JSON.parse(el.textContent);
    const hasPlotly = typeof Plotly !== "undefined";
    const S = {
      kind: "gpu",
      vendor: "all",
      alpha: 1,
      metric: "ppw",
      top: 15,
      rate: D.summary.electricity.usd_per_kwh,
      hours: D.summary.tco_defaults.hours_per_day,
      share: D.summary.tco_defaults.load_share,
      years: D.summary.tco_defaults.years,
      spVendor: "all",
    };

    function colors() {
      const cs = getComputedStyle(document.body);
      const v = (n) => cs.getPropertyValue(n).trim();
      return {
        ink: v("--vd-ink"),
        muted: v("--vd-muted"),
        grid: v("--vd-grid"),
        surface: v("--vd-surface"),
        vendor: (m) => v("--vd-s" + (SLOT[m] || 3)),
      };
    }
    function layout(c, extra) {
      return Object.assign(
        {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: c.ink, size: 12.5, family: getComputedStyle(document.body).fontFamily },
          margin: { t: 12, r: 16, b: 52, l: 64 },
          hoverlabel: { bgcolor: c.surface, bordercolor: c.grid, font: { color: c.ink, size: 12.5 } },
          showlegend: false,
          dragmode: false,
        },
        extra
      );
    }
    function axis(c, title, extra) {
      return Object.assign(
        {
          title: { text: title, font: { color: c.muted, size: 12 } },
          showgrid: true,
          gridcolor: c.grid,
          zeroline: false,
          linecolor: c.muted,
          ticks: "outside",
          tickcolor: c.grid,
          tickfont: { color: c.muted, size: 11 },
          automargin: true,
        },
        extra
      );
    }
    function empty(id, text) {
      const t = $(id);
      if (hasPlotly) Plotly.purge(t);
      t.innerHTML = '<div class="vd-state">' + esc(text) + "</div>";
    }
    function clear(id) {
      const t = $(id);
      const st = t.querySelector(".vd-state");
      if (st) t.innerHTML = "";
      return t;
    }

    function rows() {
      return D.models.filter(
        (m) => m.kind === S.kind && (S.vendor === "all" || m.vendor === S.vendor)
      );
    }
    const unit = () => (S.kind === "gpu" ? "GPU" : "CPU");

    function renderHead() {
      document.querySelectorAll("[data-ef-kind]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.efKind === S.kind))
      );
      document.querySelectorAll(".ef-kind").forEach((s) => (s.textContent = unit()));
      $("ef-title").textContent = (S.kind === "gpu" ? "Graphics card" : "Processor") + " efficiency";
      const k = D.summary.kinds[S.kind];
      if (!k) {
        $("ef-best-name").textContent = "Not collected yet";
        return;
      }
      $("ef-best-name").textContent = k.best_name;
      $("ef-best-meta").textContent =
        fmt1(k.best_score_per_watt) + " points per rated watt · " + fmtInt(k.best_watts) + " W";
      if (k.change_pct === null) {
        $("ef-trend").textContent = "Too few launches";
        $("ef-trend-note").textContent = "";
      } else {
        $("ef-trend").textContent = fmtPct(k.change_pct);
        $("ef-trend-note").textContent =
          "Median of " + k.recent_years.join("–") + " launches against " + k.before_years.join("–");
      }
      $("ef-cover").textContent = fmtInt(k.n);
      $("ef-cover-note").textContent = fmtInt(k.n_priced) + " with a current price · all power rated";
    }

    function renderVendors() {
      const sel = $("ef-vendor");
      const have = new Set(D.models.filter((m) => m.kind === S.kind).map((m) => m.vendor));
      if (S.vendor !== "all" && !have.has(S.vendor)) S.vendor = "all";
      sel.innerHTML =
        '<option value="all">All makers</option>' +
        Object.keys(VENDOR)
          .filter((v) => have.has(v))
          .map((v) => `<option value="${v}"${v === S.vendor ? " selected" : ""}>${VENDOR[v]}</option>`)
          .join("");
    }

    function renderScatter(part, c) {
      if (!part.length) return empty("ef-scatter", "No " + unit() + " has both a score and a rated power figure.");
      if (!hasPlotly) return empty("ef-scatter", "The chart library did not load.");
      const on = frontier(part);
      const traces = Object.keys(VENDOR)
        .filter((v) => part.some((m) => m.vendor === v))
        .map((v) => {
          const pts = part.filter((m) => m.vendor === v);
          return {
            type: "scatter",
            mode: "markers",
            name: VENDOR[v],
            x: pts.map((m) => m.watts),
            y: pts.map((m) => m.score),
            text: pts.map(
              (m) =>
                `<b>${esc(m.name)}</b> (${m.year})<br>${fmtInt(m.score)} points · ${fmtInt(m.watts)} W ${esc(
                  m.power_label
                )} (rated)<br>${fmt1(m.score_per_watt)} points per watt` +
                (m.price_usd ? "<br>" + fmtUsd(m.price_usd) : "")
            ),
            hovertemplate: "%{text}<extra></extra>",
            marker: { size: 9, color: "rgba(0,0,0,0)", line: { width: 1.8, color: c.vendor(v) } },
          };
        });
      const fr = part.filter((_, i) => on[i]).sort((a, b) => a.watts - b.watts);
      traces.push({
        type: "scatter",
        mode: "lines",
        x: fr.map((m) => m.watts),
        y: fr.map((m) => m.score),
        line: { color: c.muted, width: 1.5, dash: "dot", shape: "hv" },
        hoverinfo: "skip",
      });
      Plotly.react(
        clear("ef-scatter"),
        traces,
        layout(c, {
          height: 420,
          xaxis: axis(c, "Rated power (W, log scale)", { type: "log", tickvals: W_TICKS }),
          yaxis: axis(c, "Blender score (samples per minute, log scale)", { type: "log", tickvals: S_TICKS, tickformat: "," }),
        }),
        CONFIG
      );
      $("ef-legend").innerHTML =
        traces
          .slice(0, -1)
          .map((t) => `<span><span class="vx-dot vx-hollow" style="border-color:${c.vendor(
            Object.keys(VENDOR).find((k) => VENDOR[k] === t.name)
          )}"></span>${t.name}</span>`)
          .join("") + '<span><span class="ef-dash"></span>Frontier: most score for the watts</span>';
    }

    function metricValue(m) {
      return S.metric === "ppw"
        ? m.score_per_watt
        : edp(m.score, m.watts, D.summary.job_samples, S.alpha);
    }

    function renderTable(part) {
      const list = part
        .map((m) => ({ m, v: metricValue(m) }))
        .sort((a, b) => (S.metric === "ppw" ? b.v - a.v : a.v - b.v));
      const shown = S.top === "all" ? list : list.slice(0, S.top);
      const best = list.length ? list[0].v : 1;
      $("ef-alpha-wrap").hidden = S.metric === "ppw";
      $("ef-alpha-val").textContent = S.alpha.toFixed(1);
      $("ef-metric-note").textContent =
        S.metric === "ppw"
          ? "Blender points per rated watt. Higher is better."
          : "Energy × time" + (S.alpha === 1 ? "" : "^" + S.alpha.toFixed(1)) +
            " for a " + fmtInt(D.summary.job_samples) + "-sample render, as an index where the best " +
            unit() + " is 100. Lower is better.";
      if (!shown.length) {
        $("ef-table").innerHTML = '<div class="vd-state">Nothing to rank.</div>';
        return;
      }
      const head =
        "<tr><th>#</th><th>Model</th><th>Year</th><th class='num'>Score</th><th class='num'>Rated W</th>" +
        "<th class='num'>" + (S.metric === "ppw" ? "Points per W" : "EDP index") + "</th>" +
        "<th class='num'>kJ per render</th><th class='num'>Price</th></tr>";
      const body = shown
        .map(({ m, v }, i) => {
          const val = S.metric === "ppw" ? fmt1(v) : fmt1((100 * v) / best);
          return (
            `<tr><td>${i + 1}</td><td><span class="ef-swatch" style="background:var(--vd-s${SLOT[m.vendor] || 3})"></span>${esc(
              m.name
            )}${m.on_frontier ? ' <span class="ef-tag" title="On the efficiency frontier">frontier</span>' : ""}</td>` +
            `<td>${m.year}</td><td class="num">${fmtInt(m.score)}</td><td class="num">${fmtInt(m.watts)}</td>` +
            `<td class="num"><b>${val}</b></td><td class="num">${fmt1(m.energy_kj_per_job)}</td>` +
            `<td class="num">${m.price_usd ? fmtUsd(m.price_usd) : "—"}</td></tr>`
          );
        })
        .join("");
      $("ef-table").innerHTML = `<table class="ef-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
      $("ef-count").textContent =
        "Showing " + shown.length + " of " + list.length + " " + unit() + "s with a rated power figure.";
    }

    function renderTco(part, c) {
      const priced = part.filter((m) => m.price_usd > 0);
      const note = D.summary.electricity.fallback
        ? "Default rate: approximate, until the BLS series is collected."
        : "Default: BLS US city average, " + D.summary.electricity.period + ".";
      $("ef-rate-note").textContent = note;
      if (!priced.length) return empty("ef-tco", "No " + unit() + " with a rated power figure has a current price.");
      if (!hasPlotly) return empty("ef-tco", "The chart library did not load.");
      const list = priced
        .map((m) => Object.assign({ m }, tco(m.price_usd, m.watts, S.rate, S.hours, S.share, S.years)))
        .sort((a, b) => a.total - b.total)
        .slice(0, 12)
        .reverse();
      const names = list.map((r) => r.m.name);
      const tip = (r) =>
        `<b>${esc(r.m.name)}</b><br>Price ${fmtUsd(r.m.price_usd)} + electricity ${fmtUsd(r.power)}<br>Total ${fmtUsd(
          r.total
        )} over ${S.years} years`;
      Plotly.react(
        clear("ef-tco"),
        [
          {
            type: "bar", orientation: "h", name: "Price", y: names, x: list.map((r) => r.m.price_usd),
            marker: { color: c.vendor("intel") }, hovertext: list.map(tip), hovertemplate: "%{hovertext}<extra></extra>",
          },
          {
            type: "bar", orientation: "h", name: "Electricity", y: names, x: list.map((r) => r.power),
            marker: { color: c.vendor("nvidia") }, hovertext: list.map(tip), hovertemplate: "%{hovertext}<extra></extra>",
          },
        ],
        layout(c, {
          barmode: "stack",
          height: 60 + list.length * 28,
          margin: { t: 8, r: 16, b: 44, l: 8 },
          xaxis: axis(c, "Total cost over " + S.years + " years (USD)", { tickprefix: "$" }),
          yaxis: axis(c, "", { showgrid: false, ticks: "", automargin: true }),
        }),
        CONFIG
      );
      const cheapest = list[list.length - 1];
      const kwh = (S.hours * S.share * 365 * S.years) / 1000;
      $("ef-tco-summary").textContent =
        "Cheapest to own: " + cheapest.m.name + " at " + fmtUsd(cheapest.total) + ". Each rated watt adds " +
        "$" + (kwh * S.rate).toFixed(2) + " over " + S.years + " years at these settings.";
    }

    function renderTrend(c) {
      const t = D.trend.filter((r) => r.kind === S.kind).sort((a, b) => a.year - b.year);
      if (t.length < 2) {
        empty("ef-trend-chart", "Too few launch years to draw a trend.");
        empty("ef-watts-chart", "Too few launch years to draw a trend.");
        return;
      }
      if (!hasPlotly) return;
      const yrs = t.map((r) => r.year);
      const hover = (lab, f) => t.map((r) => `<b>${r.year}</b> · ${r.n} launches<br>${lab}: ${f(r)}`);
      Plotly.react(
        clear("ef-trend-chart"),
        [
          { type: "scatter", mode: "lines+markers", x: yrs, y: t.map((r) => r.best), line: { color: c.vendor("nvidia"), width: 2 },
            text: hover("Best", (r) => fmt1(r.best)), hovertemplate: "%{text}<extra></extra>" },
          { type: "scatter", mode: "lines+markers", x: yrs, y: t.map((r) => r.median), line: { color: c.vendor("intel"), width: 2 },
            text: hover("Median", (r) => fmt1(r.median)), hovertemplate: "%{text}<extra></extra>" },
        ],
        layout(c, { height: 300, xaxis: axis(c, "Launch year", { dtick: 2 }), yaxis: axis(c, "Points per rated watt", { rangemode: "tozero" }) }),
        CONFIG
      );
      Plotly.react(
        clear("ef-watts-chart"),
        [
          { type: "bar", x: yrs, y: t.map((r) => r.median_watts), marker: { color: c.vendor("amd") },
            hovertext: hover("Median rated power", (r) => fmtInt(r.median_watts) + " W"), hovertemplate: "%{hovertext}<extra></extra>" },
        ],
        layout(c, { height: 300, xaxis: axis(c, "Launch year", { dtick: 2 }), yaxis: axis(c, "Median rated power (W)", { rangemode: "tozero" }) }),
        CONFIG
      );
    }

    function renderSpecPower(c) {
      const P = D.sp.processors.filter((r) => S.spVendor === "all" || r.cpu_vendor === S.spVendor);
      const all = D.sp.processors;
      if (!all.length) {
        ["ef-sp-trend", "ef-sp-curve"].forEach((id) => empty(id, "SPECpower results have not been collected yet."));
        $("ef-sp-table").innerHTML = '<div class="vd-state">SPECpower results have not been collected yet.</div>';
        return;
      }
      const best = all[0];
      $("ef-sp-best").textContent = best.processor;
      $("ef-sp-best-note").textContent =
        fmtInt(best.ops_per_watt) + " ssj_ops per watt · median of " + best.n_results + " result" + (best.n_results > 1 ? "s" : "");
      $("ef-sp-double").textContent = D.sp.doubling_years ? D.sp.doubling_years.toFixed(1) + " years" : "Too few years";
      const t = D.sp.trend.slice().sort((a, b) => a.year - b.year);
      const last = t[t.length - 1];
      $("ef-sp-idle").textContent = last ? Math.round(100 * last.idle_share) + "%" : "—";
      $("ef-sp-idle-note").textContent = last
        ? "Median idle power as a share of full load, results published in " + last.year
        : "";
      if (hasPlotly && t.length) {
        const hover = (lab, f) => t.map((r) => `<b>${r.year}</b> · ${r.n} results<br>${lab}: ${f(r)}`);
        Plotly.react(
          clear("ef-sp-trend"),
          [
            { type: "scatter", mode: "lines+markers", x: t.map((r) => r.year), y: t.map((r) => r.best), line: { color: c.vendor("nvidia"), width: 2 },
              hovertext: hover("Best", (r) => fmtInt(r.best) + " ssj_ops/W"), hovertemplate: "%{hovertext}<extra></extra>" },
            { type: "scatter", mode: "lines+markers", x: t.map((r) => r.year), y: t.map((r) => r.median), line: { color: c.vendor("intel"), width: 2 },
              hovertext: hover("Median", (r) => fmtInt(r.median) + " ssj_ops/W"), hovertemplate: "%{hovertext}<extra></extra>" },
          ],
          layout(c, { height: 300, xaxis: axis(c, "Year published"), yaxis: axis(c, "ssj_ops per watt (log scale)", { type: "log", tickformat: ",", tickvals: [500, 1000, 2000, 5000, 10000, 20000, 50000] }) }),
          CONFIG
        );
        const loads = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        const top = P.slice(0, 5);
        const palette = [c.vendor("intel"), c.vendor("amd"), c.vendor("nvidia"), "#8a63d2", "#c9a227"];
        Plotly.react(
          clear("ef-sp-curve"),
          top.map((r, i) => ({
            type: "scatter", mode: "lines", name: r.processor, x: loads, y: loads.map((l) => r["ppw_" + l]),
            line: { color: palette[i % palette.length], width: 2 },
            hovertemplate: "<b>" + esc(r.processor) + "</b><br>%{x}% load: %{y:,.0f} ssj_ops/W<extra></extra>",
          })),
          layout(c, { height: 300, showlegend: true, legend: { orientation: "h", y: -0.3, font: { size: 11 } },
            margin: { t: 12, r: 16, b: 96, l: 64 },
            xaxis: axis(c, "Target load (%)", { dtick: 10 }), yaxis: axis(c, "ssj_ops per watt", { rangemode: "tozero", tickformat: "," }) }),
          CONFIG
        );
      }
      const rowsHtml = P.slice(0, 20).map((r, i) =>
        `<tr><td>${i + 1}</td><td>${esc(r.processor)}</td><td>${r.year}</td><td class="num">${r.n_results}</td>` +
        `<td class="num"><b>${fmtInt(r.ops_per_watt)}</b></td><td class="num">${r.idle_share == null ? "—" : Math.round(100 * r.idle_share) + "%"}</td></tr>`
      ).join("");
      $("ef-sp-table").innerHTML = P.length
        ? `<table class="ef-table"><thead><tr><th>#</th><th>Processor</th><th>First result</th><th class="num">Results</th><th class="num">ssj_ops per W</th><th class="num">Idle share</th></tr></thead><tbody>${rowsHtml}</tbody></table>`
        : '<div class="vd-state">No results for this maker.</div>';
    }


    // ---------------------------------------------------------------- models and signals
    const ML = D.ml || { residuals: [], power: [], validation: [], tiers: [], signals: [], series: [] };
    const ARROW = {
      slowing: "↘", "speeding up": "↗", steady: "→", climbing: "↗", falling: "↘", flat: "→",
      widening: "↔", closing: "→←", "no clear change": "→", "bends up": "⤴", "bends down": "⤵",
      "no bend": "→", collecting: "…",
    };
    const ORDER = ["frontier_pace", "turn_frontier", "specpower_pace", "turn_specpower", "tdp_creep",
      "turn_power", "vendor_gap", "price_cohort", "price_daily"];

    function seriesFor(signal) {
      return ML.series.filter((r) => r.signal === signal && r.kind === S.kind);
    }

    function miniChart(id, sig, c) {
      const pts = seriesFor(sig.signal);
      if (!pts.length || !hasPlotly) return false;
      const groups = [...new Set(pts.map((r) => r.group))];
      const palette = [c.vendor("intel"), c.vendor("amd"), c.vendor("nvidia"), "#8a63d2", "#c9a227",
        "#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057"];
      const logY = ["frontier_pace", "specpower_pace", "tdp_creep", "turn_frontier", "turn_specpower",
        "turn_power", "price_cohort", "price_daily"].includes(sig.signal);
      const traces = groups.map((g, i) => {
        const p = pts.filter((r) => r.group === g).sort((a, b) => a.x - b.x);
        const vendorColor = { NVIDIA: c.vendor("nvidia"), AMD: c.vendor("amd"), Intel: c.vendor("intel") }[g];
        return {
          type: sig.signal === "price_cohort" ? "bar" : "scatter",
          mode: "lines+markers",
          name: g,
          x: p.map((r) => r.x),
          y: p.map((r) => r.y),
          marker: { size: 5, color: vendorColor || palette[i % palette.length] },
          line: { width: 1.6, color: vendorColor || palette[i % palette.length] },
          hovertemplate: "<b>" + esc(g) + "</b><br>%{x}: %{y:,.3~g}<extra></extra>",
        };
      });
      const fit = pts.filter((r) => r.fit !== null).sort((a, b) => a.x - b.x);
      if (fit.length) {
        traces.push({
          type: "scatter", mode: "lines", name: sig.signal.startsWith("turn") ? "Bent line" : "Running best",
          x: fit.map((r) => r.x), y: fit.map((r) => r.fit), hoverinfo: "skip",
          line: { color: c.muted, width: 1.4, dash: "dot" },
        });
      }
      Plotly.react(
        clear(id),
        traces,
        layout(c, {
          height: 180,
          margin: { t: 6, r: 8, b: 30, l: 44 },
          showlegend: groups.length > 1 && groups.length <= 4,
          legend: { orientation: "h", y: -0.25, font: { size: 10 } },
          xaxis: axis(c, "", { tickfont: { color: c.muted, size: 10 }, title: undefined,
            dtick: sig.signal === "price_daily" ? undefined : 2 }),
          yaxis: axis(c, "", { tickfont: { color: c.muted, size: 10 }, title: undefined,
            type: logY ? "log" : "linear", tickformat: "~g", nticks: 5 }),
        }),
        CONFIG
      );
      return true;
    }

    function renderWind(c) {
      const list = ML.signals
        .filter((r) => r.kind === S.kind)
        .sort((a, b) => ORDER.indexOf(a.signal) - ORDER.indexOf(b.signal));
      const box = $("ef-wind");
      if (!list.length) {
        box.innerHTML = '<div class="vd-state">No signals yet: too few launches with a score and a rated power figure.</div>';
        return;
      }
      box.innerHTML = list
        .map(
          (r, i) =>
            `<article class="ef-sig ef-dir-${esc(r.direction).replace(/\s+/g, "-")}">` +
            `<header><h3>${esc(r.title)}</h3><span class="ef-dir">${ARROW[r.direction] || ""} ${esc(r.direction)}</span></header>` +
            `<p>${esc(r.sentence)}</p>` +
            `<div id="ef-sig-${i}" class="ef-sig-chart"></div>` +
            `<p class="vx-fine" title="${esc(r.method)}">${esc(r.method)}</p></article>`
        )
        .join("");
      list.forEach((r, i) => {
        if (!miniChart("ef-sig-" + i, r, c)) $("ef-sig-" + i).remove();
      });
    }

    function valLine(target, what) {
      const v = ML.validation.filter((r) => r.kind === S.kind && r.target === target);
      const by = Object.fromEntries(v.map((r) => [r.model, r]));
      if (!by.Linear) return "";
      const base = by["Launch year only"];
      return (
        "Typical held-out error: linear " + by.Linear.median_abs_pct_error.toFixed(1) + "%, gradient boosting " +
        by["Gradient boosting"].median_abs_pct_error.toFixed(1) + "%, against " + base.median_abs_pct_error.toFixed(1) +
        "% for launch year alone, over " + by.Linear.n_chips + " " + unit() + "s. The 80% range held the measured value for " +
        Math.round(by.Linear.range_coverage_pct) + "% of them (" + what + ")."
      );
    }

    const VERDICT_SLOT = { "Beats its specs": "nvidia", "Misses its specs": "amd", "In line with its specs": null };

    function renderResiduals(c) {
      const part = ML.residuals.filter((r) => r.kind === S.kind);
      $("ef-res-val").textContent = valLine("points per rated watt", "measured efficiency");
      if (!part.length) {
        empty("ef-res-chart", "Too few " + unit() + "s to fit the model.");
        $("ef-res-beats").innerHTML = $("ef-res-misses").innerHTML = "";
        return;
      }
      if (hasPlotly) {
        const traces = Object.keys(VERDICT_SLOT).map((v) => {
          const p = part.filter((r) => r.verdict === v);
          const col = VERDICT_SLOT[v] ? c.vendor(VERDICT_SLOT[v]) : c.muted;
          return {
            type: "scatter", mode: "markers", name: v,
            x: p.map((r) => r.expected_ppw), y: p.map((r) => r.score_per_watt),
            text: p.map((r) =>
              `<b>${esc(r.name)}</b> (${r.year}, ${esc(r.arch)}, ${esc(r.node)})<br>Measured ${fmtE(r.score_per_watt)} · predicted ${fmtE(
                r.expected_ppw)} (${fmtE(r.expected_lo)}–${fmtE(r.expected_hi)})<br>${fmtPct(r.residual_pct)} linear · ${fmtPct(
                r.gbdt_residual_pct)} boosting`),
            hovertemplate: "%{text}<extra></extra>",
            marker: { size: v === "In line with its specs" ? 6 : 9, color: v === "In line with its specs" ? "rgba(0,0,0,0)" : col,
              line: { width: 1.4, color: col } },
          };
        });
        const all = part.flatMap((r) => [r.expected_ppw, r.score_per_watt]);
        const lo = Math.min(...all) * 0.9, hi = Math.max(...all) * 1.1;
        traces.push({ type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], hoverinfo: "skip",
          line: { color: c.muted, width: 1, dash: "dot" } });
        const ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50];
        Plotly.react(
          clear("ef-res-chart"),
          traces,
          layout(c, {
            height: 380,
            xaxis: axis(c, "Predicted points per rated watt (log)", { type: "log", tickvals: ticks }),
            yaxis: axis(c, "Measured points per rated watt (log)", { type: "log", tickvals: ticks }),
          }),
          CONFIG
        );
        $("ef-res-legend").innerHTML = Object.keys(VERDICT_SLOT).map((v) =>
          `<span><span class="vx-dot${VERDICT_SLOT[v] ? "" : " vx-hollow"}" style="${VERDICT_SLOT[v]
            ? "background:" + c.vendor(VERDICT_SLOT[v]) : "border-color:" + c.muted}"></span>${v}</span>`
        ).join("") + '<span><span class="ef-dash"></span>Measured = predicted</span>';
      }
      const table = (list) =>
        list.length
          ? `<table class="ef-table"><thead><tr><th>Model</th><th>Node</th><th class="num">Measured</th><th class="num">Predicted (80% range)</th><th class="num">Gap</th></tr></thead><tbody>` +
            list.slice(0, 10).map((r) =>
              `<tr><td>${esc(r.name)}<br><span class="vx-fine">${r.year} · ${esc(r.arch)} · ${r.n_runs} runs</span></td><td>${esc(r.node)}</td>` +
              `<td class="num">${fmtE(r.score_per_watt)}</td><td class="num">${fmtE(r.expected_ppw)} <span class="vx-fine">(${fmtE(
                r.expected_lo)}–${fmtE(r.expected_hi)})</span></td><td class="num"><b>${fmtPct(r.residual_pct)}</b></td></tr>`
            ).join("") + "</tbody></table>"
          : '<div class="vd-state">None.</div>';
      $("ef-res-beats").innerHTML = table(part.filter((r) => r.verdict === "Beats its specs"));
      $("ef-res-misses").innerHTML = table(
        part.filter((r) => r.verdict === "Misses its specs").sort((a, b) => a.residual_pct - b.residual_pct)
      );
    }

    function renderTiers(c) {
      const summ = ML.tiers.filter((r) => r.kind === S.kind).sort((a, b) => a.tier - b.tier);
      const chips = ML.residuals.filter((r) => r.kind === S.kind && r.tier !== null);
      if (!summ.length) {
        $("ef-tier-note").textContent = "";
        $("ef-tier-table").innerHTML = "";
        return empty("ef-tier-chart", "Too few " + unit() + "s in today's line-up to look for tiers.");
      }
      const k = summ[0].n_tiers;
      $("ef-tier-note").textContent =
        k === 1
          ? "No natural tiers: today's " + unit() + "s form one continuous band of efficiency, from " +
            fmtE(summ[0].min_ppw) + " to " + fmtE(summ[0].max_ppw) + " points per rated watt. Any bracket would be drawn by hand."
          : "The data supports " + k + " tiers among " + chips.length + " " + unit() + "s" +
            (summ[0].clear ? "." : ", though other tier counts fit almost as well, so treat the boundaries as soft.") +
            " Borderline chips sit between two tiers.";
      if (hasPlotly && chips.length) {
        const palette = [c.vendor("nvidia"), c.vendor("intel"), c.vendor("amd"), "#8a63d2", "#c9a227", "#d1495b"];
        Plotly.react(
          clear("ef-tier-chart"),
          summ.map((t) => {
            const p = chips.filter((r) => r.tier === t.tier);
            return {
              type: "scatter", mode: "markers", name: "Tier " + t.tier,
              x: p.map((r) => r.score_per_watt), y: p.map(() => "Tier " + t.tier),
              text: p.map((r) => `<b>${esc(r.name)}</b> (${r.year})<br>${fmtE(r.score_per_watt)} points per rated watt<br>Tier ${t.tier}, ${Math.round(
                100 * r.tier_prob)}% membership${r.borderline ? " · borderline" : ""}`),
              hovertemplate: "%{text}<extra></extra>",
              marker: { size: 9, color: p.map((r) => (r.borderline ? "rgba(0,0,0,0)" : palette[(t.tier - 1) % palette.length])),
                line: { width: 1.5, color: palette[(t.tier - 1) % palette.length] } },
            };
          }),
          layout(c, {
            height: 90 + 46 * summ.length,
            margin: { t: 8, r: 16, b: 44, l: 64 },
            xaxis: axis(c, "Points per rated watt (log)", { type: "log", tickvals: [0.5, 1, 2, 5, 10, 20, 50] }),
            yaxis: axis(c, "", { autorange: "reversed", showgrid: false }),
          }),
          CONFIG
        );
      }
      $("ef-tier-table").innerHTML =
        `<table class="ef-table"><thead><tr><th>Tier</th><th class="num">Chips</th><th class="num">Points per rated watt</th><th class="num">Median launch</th><th>Most efficient</th></tr></thead><tbody>` +
        summ.map((t) => `<tr><td>Tier ${t.tier}</td><td class="num">${t.n}</td><td class="num">${fmtE(t.min_ppw)}–${fmtE(t.max_ppw)}</td>` +
          `<td class="num">${Math.round(t.median_year)}</td><td>${esc(t.examples)}</td></tr>`).join("") + "</tbody></table>";
    }

    function renderGaps() {
      const part = ML.power.filter((r) => r.kind === S.kind);
      $("ef-gap-val").textContent = valLine("rated power", "rated power");
      const est = part.filter((r) => r.status === "Estimated");
      $("ef-gap-table").innerHTML = est.length
        ? `<table class="ef-table"><thead><tr><th>Model</th><th>Node</th><th class="num">Estimated W (80% range)</th><th class="num">Points per estimated W</th></tr></thead><tbody>` +
          est.map((r) => `<tr><td>${esc(r.name)}<br><span class="vx-fine">${r.year} · ${esc(r.arch)} · ${esc(r.model)} model</span></td><td>${esc(r.node)}</td>` +
            `<td class="num"><b>${fmtInt(r.estimate_w)}</b> <span class="vx-fine">(${fmtInt(r.estimate_lo_w)}–${fmtInt(r.estimate_hi_w)})</span></td>` +
            `<td class="num">${fmt1(r.est_score_per_watt)}</td></tr>`).join("") + "</tbody></table>"
        : '<div class="vd-state">Every benchmarked ' + unit() + " with specifications already has a power figure.</div>";
      const odd = part.filter((r) => r.status === "Held-out check");
      $("ef-gap-odd").innerHTML = odd.length
        ? `<table class="ef-table"><thead><tr><th>Model</th><th class="num">Rated W</th><th class="num">Specs predict (80% range)</th><th class="num">Rated vs predicted</th></tr></thead><tbody>` +
          odd.map((r) => `<tr><td>${esc(r.name)}<br><span class="vx-fine">${r.year} · ${esc(r.arch)}</span></td><td class="num">${fmtInt(r.rated_w)}</td>` +
            `<td class="num">${fmtInt(r.estimate_w)} <span class="vx-fine">(${fmtInt(r.estimate_lo_w)}–${fmtInt(r.estimate_hi_w)})</span></td>` +
            `<td class="num"><b>${fmtPct(r.gap_pct)}</b></td></tr>`).join("") + "</tbody></table>"
        : "";
    }

    function render() {
      const c = colors();
      document.querySelectorAll("[data-ef-for]").forEach((s) => (s.hidden = s.dataset.efFor !== S.kind));
      renderHead();
      renderVendors();
      const part = rows();
      renderScatter(part, c);
      renderTable(part);
      renderTco(part, c);
      renderTrend(c);
      renderWind(c);
      renderResiduals(c);
      renderTiers(c);
      renderGaps();
      if (S.kind === "cpu") renderSpecPower(c);
    }

    // ---------------------------------------------------------------- events
    document.querySelectorAll("[data-ef-kind]").forEach((b) =>
      b.addEventListener("click", () => {
        S.kind = b.dataset.efKind;
        render();
      })
    );
    $("ef-sp-vendor").addEventListener("change", (e) => {
      S.spVendor = e.target.value;
      renderSpecPower(colors());
    });
    $("ef-vendor").addEventListener("change", (e) => {
      S.vendor = e.target.value;
      render();
    });
    document.querySelectorAll("[data-ef-metric]").forEach((b) =>
      b.addEventListener("click", () => {
        S.metric = b.dataset.efMetric;
        document.querySelectorAll("[data-ef-metric]").forEach((x) =>
          x.setAttribute("aria-pressed", String(x === b))
        );
        renderTable(rows());
      })
    );
    $("ef-alpha").addEventListener("input", (e) => {
      S.alpha = Number(e.target.value);
      renderTable(rows());
    });
    $("ef-top").addEventListener("change", (e) => {
      S.top = e.target.value === "all" ? "all" : Number(e.target.value);
      renderTable(rows());
    });
    [["ef-rate", "rate"], ["ef-hours", "hours"], ["ef-share", "share", 0.01], ["ef-years", "years"]].forEach(
      ([id, key, scale]) => {
        const input = $(id);
        input.value = scale ? Math.round(S[key] / scale) : S[key];
        input.addEventListener("input", () => {
          const v = Number(input.value);
          if (!Number.isFinite(v) || v < 0) return;
          S[key] = scale ? v * scale : v;
          renderTco(rows(), colors());
        });
      }
    );

    let pending = null;
    new MutationObserver(() => {
      clearTimeout(pending);
      pending = setTimeout(render, 120);
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    render();
    window.addEventListener("load", () => setTimeout(render, 120));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(typeof window !== "undefined" ? window : this);
