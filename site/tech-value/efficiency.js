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

    function render() {
      const c = colors();
      renderHead();
      renderVendors();
      const part = rows();
      renderScatter(part, c);
      renderTable(part);
      renderTco(part, c);
      renderTrend(c);
    }

    // ---------------------------------------------------------------- events
    document.querySelectorAll("[data-ef-kind]").forEach((b) =>
      b.addEventListener("click", () => {
        S.kind = b.dataset.efKind;
        render();
      })
    );
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
