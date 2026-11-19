// Value ranking dashboard: filters, linked selection and charts for site/tech-value/index.qmd.
//
// The page embeds the data as JSON (built by pdi/tech_value/dashboard.py) and this script
// draws it with Plotly. The calculations at the top mirror the Python ones and are checked
// against them by tests/test_value_dashboard.py, which runs this file under Node.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- calculations
  function quantile(values, q) {
    const a = values.filter((v) => v !== null && v !== undefined).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const pos = (a.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }

  // Share of the other models with a strictly lower value.
  function shareBelow(values, x) {
    if (values.length < 2) return null;
    return values.filter((v) => v < x).length / (values.length - 1);
  }

  function dominates(a, b) {
    return (
      a.price_usd <= b.price_usd &&
      a.score >= b.score &&
      (a.price_usd < b.price_usd || a.score > b.score)
    );
  }

  function frontier(rows) {
    return rows.map((r) => !rows.some((o) => o !== r && dominates(o, r)));
  }

  function bestDominator(rows, key) {
    const me = rows.find((r) => r.model_key === key);
    if (!me) return null;
    const rivals = rows.filter((o) => o.model_key !== key && dominates(o, me));
    if (!rivals.length) return null;
    const best = rivals.reduce((b, o) =>
      o.score / o.price_usd > b.score / b.price_usd ? o : b
    );
    return {
      model_key: best.model_key,
      more_score_pct: 100 * (best.score / me.score - 1),
      lower_price_pct: 100 * (1 - best.price_usd / me.price_usd),
    };
  }

  function upgrade(a, b) {
    if (a.kind !== b.kind) throw new Error("compare models of the same kind only");
    return {
      price_diff_usd: b.price_usd - a.price_usd,
      price_pct: 100 * (b.price_usd / a.price_usd - 1),
      score_diff: b.score - a.score,
      score_pct: 100 * (b.score / a.score - 1),
      value_pct: 100 * (b.score_per_dollar / a.score_per_dollar - 1),
    };
  }

  function median(values) {
    return quantile(values, 0.5);
  }

  // [lo, hi) brackets from sorted positive edges, the same rule as the Python version.
  function brackets(rows, edges) {
    const e = edges.filter((x) => x > 0).sort((a, b) => a - b);
    const bounds = [0, ...e, Infinity];
    const out = [];
    for (let i = 0; i < bounds.length - 1; i++) {
      const lo = bounds[i];
      const hi = bounds[i + 1];
      const part = rows.filter((r) => r.price_usd >= lo && r.price_usd < hi);
      const row = { low: lo, high: hi, n: part.length };
      if (part.length) {
        const best = part.reduce((b, r) => (r.score_per_dollar > b.score_per_dollar ? r : b));
        const top = part.reduce((b, r) => (r.score > b.score ? r : b));
        Object.assign(row, {
          best_value: best.model_key,
          best_value_ratio: best.score_per_dollar,
          top_score_model: top.model_key,
          top_score: top.score,
          median_score: median(part.map((r) => r.score)),
          median_price: median(part.map((r) => r.price_usd)),
        });
      }
      out.push(row);
    }
    return out;
  }

  const MAKERS = { nvidia: "NVIDIA", amd: "AMD", intel: "Intel" };
  const UPPER = new Set(["rtx", "rx", "gtx", "xt", "xtx", "gre", "ai"]);
  function displayName(key) {
    return String(key)
      .split(" ")
      .map((w) => {
        if (MAKERS[w]) return MAKERS[w];
        if (UPPER.has(w)) return w.toUpperCase();
        if (w === "ti") return "Ti";
        if (/\d/.test(w)) {
          const up = w.toUpperCase();
          return up.length > 1 && up[0] === "I" && /\d/.test(up[1]) ? "i" + up.slice(1) : up;
        }
        return w.charAt(0).toUpperCase() + w.slice(1);
      })
      .join(" ");
  }

  const calc = { quantile, shareBelow, dominates, frontier, bestDominator, upgrade, brackets, displayName };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = calc;
    return;
  }
  root.VDcalc = calc;

  // ---------------------------------------------------------------- page
  const $ = (id) => document.getElementById(id);
  const fmtUsd = (x) => "$" + Math.round(x).toLocaleString("en-US");
  const fmtUsd2 = (x) =>
    "$" + x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
  const fmtPct = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(0) + "%";
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const KSHORT = { gpu: "GPU", cpu: "CPU" };
  const MAKER_SLOT = { NVIDIA: 1, AMD: 2, Intel: 3 };
  const LOG_TICKS = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000];

  function boot() {
    const dataEl = $("vd-data");
    if (!dataEl) return;
    const D = JSON.parse(dataEl.textContent);
    let changeChartKind;
    const switches = document.querySelectorAll("[data-vd-kind]");
    switches.forEach((button) => button.addEventListener("click", () => {
      const kind = button.dataset.vdKind;
      switches.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      document.querySelectorAll("[data-insight-kind]").forEach((panel) => {
        panel.hidden = panel.dataset.insightKind !== kind;
      });
      if (changeChartKind) changeChartKind(kind);
    }));
    if (typeof Plotly === "undefined") {
      document.querySelectorAll(".vd-chart").forEach((el) => {
        el.innerHTML =
          '<div class="vd-state vd-error">The chart library did not load, so the charts cannot be drawn. The tables further down still work.</div>';
      });
      return;
    }
    const models = D.models.map((m) => Object.assign({}, m, { name: displayName(m.model_key) }));
    const S = {
      kind: "gpu",
      makers: {},
      priceMin: null,
      priceMax: null,
      scoreMin: null,
      scoreMax: null,
      search: "",
      hideLow: false,
      medians: true,
      showFrontier: true,
      logPrice: true,
      selected: {},
      compare: {},
      top: 10,
      order: "desc",
      budget: {},
      edges: {},
    };
    for (const k of ["gpu", "cpu"]) {
      const part = models.filter((m) => m.kind === k);
      S.makers[k] = new Set(part.map((m) => m.maker));
      const normal = part.filter((m) => m.confidence !== "low");
      const pool = normal.length ? normal : part;
      const best = pool.reduce((b, m) => (!b || m.score_per_dollar > b.score_per_dollar ? m : b), null);
      S.selected[k] = best ? best.model_key : null;
      S.edges[k] = (D.brackets[k] || []).join(", ");
      S.budget[k] = best ? Math.round(quantile(part.map((m) => m.price_usd), 0.5)) : null;
      // A product picked on another Tech Value page (or in a shared link) wins over the default.
      const shared = root.TVPick && root.TVPick.get(k);
      if (shared && part.some((m) => m.model_key === shared)) S.selected[k] = shared;
    }

    function colors() {
      const cs = getComputedStyle(document.body);
      const v = (n) => cs.getPropertyValue(n).trim();
      return {
        ink: v("--vd-ink"),
        muted: v("--vd-muted"),
        grid: v("--vd-grid"),
        surface: v("--vd-surface"),
        accent: v("--vd-accent"),
        quad: v("--vd-quad"),
        quadPoor: v("--vd-quad-poor") || "rgba(217, 89, 38, 0.05)",
        maker: (m) => v("--vd-s" + (MAKER_SLOT[m] || 1)),
      };
    }

    function allOfKind() {
      return models.filter((m) => m.kind === S.kind);
    }

    function filtered() {
      const q = S.search.trim().toLowerCase();
      return allOfKind().filter(
        (m) =>
          S.makers[S.kind].has(m.maker) &&
          (S.priceMin === null || m.price_usd >= S.priceMin) &&
          (S.priceMax === null || m.price_usd <= S.priceMax) &&
          (S.scoreMin === null || m.score >= S.scoreMin) &&
          (S.scoreMax === null || m.score <= S.scoreMax) &&
          (!S.hideLow || m.confidence !== "low") &&
          (!q || m.model_key.includes(q) || m.name.toLowerCase().includes(q))
      );
    }

    function selectedModel(rows) {
      return rows.find((m) => m.model_key === S.selected[S.kind]) || null;
    }

    function layout(c, extra) {
      return Object.assign(
        {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: c.ink, size: 13, family: getComputedStyle(document.body).fontFamily },
          margin: { t: 22, r: 20, b: 56, l: 72 },
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
          showgrid: false,
          zerolinecolor: c.grid,
          linecolor: c.grid,
          tickfont: { color: c.muted, size: 11.5 },
          automargin: true,
        },
        extra
      );
    }
    const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };

    function emptyState(id, text) {
      const el = $(id);
      Plotly.purge(el);
      el.innerHTML = '<div class="vd-state">' + esc(text) + "</div>";
    }
    function ready(id) {
      const el = $(id);
      const st = el.querySelector(".vd-state");
      if (st) el.innerHTML = "";
      return el;
    }

    function hoverText(m) {
      const medLine = m.median_price_usd ? "Median listing: " + fmtUsd2(m.median_price_usd) + "<br>" : "";
      const stores = m.n_retailers ? " across " + m.n_retailers + " store" + (m.n_retailers === 1 ? "" : "s") : "";
      return (
        "<b>" + esc(m.name) + "</b><br>" +
        "Maker: " + esc(m.maker) + (m.backend ? " · " + esc(m.backend) : "") + "<br>" +
        "Lowest new listing: <b>" + fmtUsd2(m.price_usd) + "</b><br>" +
        medLine +
        "Blender score: <b>" + fmtInt(m.score) + "</b> pts<br>" +
        "Score per $: <b>" + m.score_per_dollar.toFixed(2) + "</b> pts/USD<br>" +
        "Benchmark runs: " + fmtInt(m.n_runs) + "<br>" +
        "Listings: " + (m.n_listings || 1) + stores + " (" + (m.confidence === "low" ? "single listing" : "multi-listing") + ")<br>" +
        "Collected: " + esc(m.collected_date || "")
      );
    }

    // ------------------------------------------------------------ controls
    function renderControls(rows) {
      const kindBtns = document.querySelectorAll("[data-vd-kind]");
      kindBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.vdKind === S.kind)));
      const all = allOfKind();
      const makers = [...new Set(all.map((m) => m.maker))].sort();
      $("vd-makers").innerHTML = makers
        .map(
          (mk) =>
            '<label class="vd-check"><input type="checkbox" data-vd-maker="' + esc(mk) + '"' +
            (S.makers[S.kind].has(mk) ? " checked" : "") + '><span class="vd-swatch vd-s' +
            (MAKER_SLOT[mk] || 1) + '"></span>' + esc(mk) + "</label>"
        )
        .join("");
      $("vd-count").textContent =
        "Showing " + rows.length + " of " + all.length + " ranked " + KSHORT[S.kind] + "s";
      const sel = $("vd-select");
      const sorted = rows.slice().sort((a, b) => a.name.localeCompare(b.name));
      sel.innerHTML = sorted
        .map((m) => '<option value="' + esc(m.model_key) + '"' + (m.model_key === S.selected[S.kind] ? " selected" : "") + ">" + esc(m.name) + " (" + fmtUsd(m.price_usd) + " · " + m.score_per_dollar.toFixed(1) + " pts/$)</option>")
        .join("");
      if (!rows.length) sel.innerHTML = "<option>No models match the filters</option>";
      const cmp = $("vd-compare");
      const base = selectedModel(rows);
      const others = sorted.filter((m) => !base || m.model_key !== base.model_key);
      if (!others.some((m) => m.model_key === S.compare[S.kind])) {
        // Default comparison: the next model up in price, else the next one down.
        // Prefer the next model up on the efficiency frontier: the upgrade worth weighing.
        const onF = frontier(rows);
        const eff = rows.filter((m, i) => onF[i]);
        const pickUp = (set) => (base ? set.filter((m) => m !== base && m.price_usd > base.price_usd).sort((a, b) => a.price_usd - b.price_usd)[0] : null);
        const up = pickUp(eff) || pickUp(others);
        S.compare[S.kind] = (up || others[0] || {}).model_key || null;
      }
      cmp.innerHTML = others
        .map((m) => '<option value="' + esc(m.model_key) + '"' + (m.model_key === S.compare[S.kind] ? " selected" : "") + ">" + esc(m.name) + " (" + fmtUsd(m.price_usd) + ")</option>")
        .join("");
      document.querySelectorAll("[data-vd-top]").forEach((b) => b.setAttribute("aria-pressed", String(String(S.top) === b.dataset.vdTop)));
      document.querySelectorAll("[data-vd-order]").forEach((b) => b.setAttribute("aria-pressed", String(S.order === b.dataset.vdOrder)));
      $("vd-medians").checked = S.medians;
      $("vd-frontier").checked = S.showFrontier;
      $("vd-logx").checked = S.logPrice;
      $("vd-budget").value = S.budget[S.kind] ?? "";
      $("vd-edges").value = S.edges[S.kind];
      document.querySelectorAll(".vd-kind-label").forEach((el) => (el.textContent = KSHORT[S.kind] + "s"));
    }

    // ------------------------------------------------------------ selected model card
    function renderDetails(rows) {
      const m = selectedModel(rows);
      const el = $("vd-details");
      if (!m) {
        el.innerHTML = '<div class="vd-state">Pick a model to see its details. The current filters hide the one that was selected.</div>';
        return;
      }
      const rk = rows.slice().sort((a, b) => b.score_per_dollar - a.score_per_dollar).findIndex((r) => r === m) + 1;
      const n = rows.length;
      const gate = D.gates.percentiles;
      const pct = n >= gate ? shareBelow(rows.map((r) => r.score), m.score) : null;
      const dom = bestDominator(rows, m.model_key);
      const onF = frontier(rows)[rows.indexOf(m)];
      const link = m.listing_url ? '<a class="vd-pill-link" href="' + esc(m.listing_url) + '" target="_blank" rel="noopener">View listing ↗</a>' : "";
      const slot = MAKER_SLOT[m.maker] || 1;
      const badgeFrontier = onF
        ? '<span class="vd-badge vd-badge-ok">★ Efficiency Frontier</span>'
        : '<span class="vd-badge vd-badge-dom">Dominated</span>';
      const badgeConf = m.confidence === "low"
        ? '<span class="vd-badge vd-badge-warn">Single listing</span>'
        : '<span class="vd-badge">' + (m.n_listings || 2) + " listings" + (m.n_retailers ? " · " + m.n_retailers + " stores" : "") + "</span>";
      const medSub = m.median_price_usd ? '<span class="vd-muted">Median: ' + fmtUsd2(m.median_price_usd) + "</span>" : "";
      const pctBar = pct !== null
        ? '<div class="vd-mini-bar"><span style="width:' + Math.round(100 * pct) + '%"></span></div>'
        : "";
      el.innerHTML =
        '<div class="vd-detail-head">' +
          '<div class="vd-detail-title-row">' +
            '<span class="vd-swatch vd-s' + slot + '"></span>' +
            "<strong>" + esc(m.name) + "</strong>" +
            '<span class="vd-badge">' + esc(m.maker) + (m.backend ? " · " + esc(m.backend) : "") + "</span>" +
            badgeFrontier + badgeConf +
          "</div>" +
          link +
        "</div>" +
        '<dl class="vd-dl">' +
        "<div><dt>Lowest new listing</dt><dd><strong>" + fmtUsd2(m.price_usd) + "</strong> " + medSub + "</dd></div>" +
        "<div><dt>Score per dollar</dt><dd><strong>" + m.score_per_dollar.toFixed(2) + ' pts/$</strong> <span class="vd-muted">Rank #' + rk + " of " + n + "</span></dd></div>" +
        "<div><dt>Blender score</dt><dd><strong>" + fmtInt(m.score) + '</strong> <span class="vd-muted">IQR ' + fmtInt(m.p25_score) + "–" + fmtInt(m.p75_score) + "</span></dd></div>" +
        "<div><dt>Benchmark sample</dt><dd><strong>" + fmtInt(m.n_runs) + '</strong> <span class="vd-muted">runs · collected ' + esc(m.collected_date || "") + "</span></dd></div>" +
        "<div><dt>Performance percentile</dt><dd>" +
        (pct === null
          ? '<span class="vd-muted">Needs ' + gate + " models in view (" + n + " shown)</span>"
          : "<strong>Faster than " + Math.round(100 * pct) + "%</strong> <span class=\"vd-muted\">of " + (n - 1) + " peers</span>" + pctBar) +
        "</dd></div>" +
        "<div><dt>Frontier status</dt><dd>" +
        (dom
          ? 'Beat by <strong>' + esc(displayName(dom.model_key)) + "</strong> (+" + dom.more_score_pct.toFixed(0) + "% score, −" + dom.lower_price_pct.toFixed(0) + "% price)"
          : onF
          ? '<span class="vd-ok-text">Undominated — no peer is both cheaper and faster</span>'
          : "") +
        "</dd></div>" +
        "</dl>";
    }

    // ------------------------------------------------------------ 1. performance vs price
    function renderScatter(rows, c) {
      if (!rows.length) {
        $("vd-scatter-n").textContent = "";
        $("vd-scatter-legend").replaceChildren();
        return emptyState("vd-scatter", "No models match the filters.");
      }
      const el = ready("vd-scatter");
      const onF = frontier(rows);
      const best = rows.reduce((b, m) => (m.score_per_dollar > b.score_per_dollar ? m : b));
      const top = rows.reduce((b, m) => (m.score > b.score ? m : b));
      const sel = selectedModel(rows);
      const makers = [...new Set(rows.map((m) => m.maker))].sort();
      const traces = makers.map((mk) => {
        const part = rows.filter((m) => m.maker === mk);
        return {
          type: "scatter",
          mode: "markers",
          name: mk,
          x: part.map((m) => m.price_usd),
          y: part.map((m) => m.score),
          customdata: part.map((m) => m.model_key),
          text: part.map(hoverText),
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: part.map((m) => (sel && m === sel ? 17 : onF[rows.indexOf(m)] ? 12 : 10)),
            color: c.maker(mk),
            symbol: part.map((m) => (m.confidence === "low" ? "circle-open" : "circle")),
            opacity: part.map((m) => (S.showFrontier && !onF[rows.indexOf(m)] && (!sel || m !== sel) ? 0.38 : 0.95)),
            line: {
              width: part.map((m) => (sel && m === sel ? 3.2 : m.confidence === "low" ? 2.2 : 1.5)),
              color: part.map((m) => (sel && m === sel ? c.ink : m.confidence === "low" ? c.maker(mk) : c.surface)),
            },
          },
        };
      });
      const shapes = [];
      const annotations = [];
      const xs = rows.map((m) => m.price_usd);
      const ys = rows.map((m) => m.score);
      const medX = median(xs);
      const medY = median(ys);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      if (S.medians && rows.length >= 2) {
        // Quadrant tint: "excellent value" (top-left) and "poor value" (bottom-right).
        shapes.push(
          { type: "rect", xref: "x", yref: "y", x0: S.logPrice ? minX / 1.35 : 0, x1: medX, y0: medY, y1: maxY * 1.12, fillcolor: "rgba(64, 180, 112, 0.17)", line: { width: 0 }, layer: "below" },
          { type: "line", xref: "x", yref: "paper", x0: medX, x1: medX, y0: 0, y1: 1, line: { color: c.muted, width: 1.2, dash: "dot" } },
          { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: medY, y1: medY, line: { color: c.muted, width: 1.2, dash: "dot" } }
        );
        const q = (x, y, t, xa, ya) => annotations.push({ xref: "paper", yref: "paper", x, y, text: t, showarrow: false, xanchor: xa, yanchor: ya, font: { size: 11, color: c.muted } });
        if (el.clientWidth >= 560) {
          q(0.015, 0.985, "<b>↗ Excellent value</b> · faster & cheaper than median", "left", "top");
          q(0.985, 0.985, "<b>Premium</b> · faster & pricier", "right", "top");
          q(0.015, 0.015, "<b>Budget</b> · slower & cheaper", "left", "bottom");
          q(0.985, 0.015, "<b>Poor value</b> · slower & pricier", "right", "bottom");
        }
        annotations.push({ xref: "x", yref: "paper", x: S.logPrice ? Math.log10(medX) : medX, y: 1.01, text: "Median price " + fmtUsd(medX), showarrow: false, yanchor: "bottom", font: { size: 10.5, color: c.muted } });
      }
      if (S.showFrontier) {
        const eff = rows.filter((m, i) => onF[i]).sort((a, b) => a.price_usd - b.price_usd);
        traces.unshift({
          type: "scatter",
          mode: "lines",
          name: "Efficiency frontier",
          x: eff.map((m) => m.price_usd),
          y: eff.map((m) => m.score),
          line: { shape: "hv", color: c.ink, width: 2, dash: "dash" },
          opacity: 0.55,
          hoverinfo: "skip",
        });
      }
      const label = (m, t, ax, ay) =>
        annotations.push({
          x: S.logPrice ? Math.log10(m.price_usd) : m.price_usd,
          y: m.score,
          text: "<b>" + esc(t) + "</b>",
          showarrow: true,
          arrowhead: 2,
          arrowsize: 0.8,
          arrowwidth: 1.2,
          arrowcolor: c.muted,
          ax,
          ay,
          font: { size: 11, color: c.ink },
          bgcolor: c.surface,
          bordercolor: c.grid,
          borderpad: 3,
          borderwidth: 1,
        });
      if (sel) {
        label(sel, "Selected", 0, -32);
      }
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: 500,
          shapes,
          annotations,
          xaxis: axis(c, "Lowest new listing price (USD" + (S.logPrice ? ", log scale)" : ")"), S.logPrice
            ? { type: "log", tickprefix: "$", separatethousands: true, tickvals: LOG_TICKS, ticktext: LOG_TICKS.map((v) => "$" + v.toLocaleString("en-US")), range: [Math.log10(minX / 1.25), Math.log10(maxX * 1.28)] }
            : { tickprefix: "$", separatethousands: true, rangemode: "tozero" }),
          yaxis: axis(c, "Blender benchmark score (points)", { separatethousands: true, range: [0, maxY * 1.12] }),
        }),
        CONFIG
      );
      bindClick(el);
      $("vd-scatter-legend").innerHTML =
        makers.map((mk) => '<span><span class="vd-swatch vd-s' + (MAKER_SLOT[mk] || 1) + '"></span>' + esc(mk) + "</span>").join("") +
        '<span><span class="vd-swatch vd-hollow"></span>Single-listing price</span>' +
        (S.showFrontier ? '<span><span class="vd-dash"></span>Efficiency frontier (faded dots are dominated)</span>' : "");
      $("vd-scatter-n").textContent = "Best value: " + best.name + " (" + fmtUsd(best.price_usd) + ") · Highest score: " + top.name + " (" + fmtInt(top.score) + " pts) · n = " + rows.length + " ranked " + KSHORT[S.kind] + "s" + (S.medians ? ". Medians: " + fmtUsd(medX) + " and " + fmtInt(medY) + " points." : ".");
    }

    function bindClick(el) {
      if (el.dataset.vdBound) return;
      el.dataset.vdBound = "1";
      el.on("plotly_click", (ev) => {
        const p = ev.points && ev.points[0];
        const key = p && (p.customdata || (p.data.customdata && p.data.customdata[p.pointIndex]));
        if (key && models.some((m) => m.model_key === key)) {
          S.selected[S.kind] = key;
          render();
        }
      });
    }

    // ------------------------------------------------------------ 2. leaderboard
    function renderBars(rows, c) {
      if (!rows.length) return emptyState("vd-bars", "No models match the filters.");
      const el = ready("vd-bars");
      let sorted = rows.slice().sort((a, b) => b.score_per_dollar - a.score_per_dollar);
      const bestKey = sorted[0].model_key;
      const medSpd = median(rows.map((m) => m.score_per_dollar));
      if (S.order === "asc") sorted.reverse();
      if (S.top !== "all") sorted = sorted.slice(0, S.top);
      const sel = selectedModel(rows);
      const shown = sorted.slice().reverse(); // Plotly draws the first category at the bottom
      const maxX = Math.max(...shown.map((m) => m.score_per_dollar), medSpd || 1);
      Plotly.react(
        el,
        [
          {
            type: "bar",
            orientation: "h",
            x: shown.map((m) => m.score_per_dollar),
            y: shown.map((m) => m.name),
            customdata: shown.map((m) => m.model_key),
            text: shown.map((m) => m.score_per_dollar.toFixed(2) + "  ·  " + fmtUsd(m.price_usd)),
            textposition: "outside",
            cliponaxis: false,
            textfont: { color: c.ink, size: 11 },
            hovertext: shown.map(hoverText),
            hovertemplate: "%{hovertext}<extra></extra>",
            marker: {
              color: shown.map((m) => c.maker(m.maker)),
              opacity: shown.map((m) => (m.confidence === "low" ? 0.45 : 0.92)),
              line: { width: shown.map((m) => (m.model_key === bestKey || m === sel ? 2.2 : 0)), color: c.ink },
            },
          },
        ],
        layout(c, {
          height: Math.max(250, 27 * shown.length + 84),
          margin: { t: 24, r: 112, b: 48, l: 8 },
          bargap: 0.28,
          shapes: medSpd
            ? [{ type: "line", xref: "x", yref: "paper", x0: medSpd, x1: medSpd, y0: 0, y1: 1, line: { color: c.muted, width: 1.2, dash: "dot" } }]
            : [],
          annotations: medSpd
            ? [{ xref: "x", yref: "paper", x: medSpd, y: 1.01, yanchor: "bottom", showarrow: false, text: "Median " + medSpd.toFixed(2) + " pts/$", font: { size: 10.5, color: c.muted } }]
            : [],
          xaxis: axis(c, "Score per dollar (Blender points per USD)", { range: [0, maxX * 1.22] }),
          yaxis: axis(c, "", { automargin: true, gridcolor: "rgba(0,0,0,0)", tickfont: { color: c.ink, size: 11.5 } }),
        }),
        CONFIG
      );
      bindClick(el);
      $("vd-bars-n").textContent = "Showing " + shown.length + " of " + rows.length + ". Bar labels show score per $ and lowest price. Outlined: best value" + (sel ? " and selected model" : "") + ". Faded: single-listing price.";
    }

    // ------------------------------------------------------------ 3. price premium vs performance gain
    function renderPremium(rows, c) {
      const base = selectedModel(rows);
      if (!base || rows.length < 2) {
        $("vd-premium-summary").innerHTML = "";
        emptyState("vd-premium-full", "Full range needs at least two models.");
        return emptyState("vd-premium", base ? "Only one model is in view; there is nothing to compare it with." : "Pick a baseline model above.");
      }
      const el = ready("vd-premium");
      const others = rows.filter((m) => m !== base);
      const pts = others.map((m) => Object.assign({ m }, upgrade(base, m)));
      const cmp = others.find((m) => m.model_key === S.compare[S.kind]);
      const xs = pts.map((p) => p.price_pct);
      const ys = pts.map((p) => p.score_pct);
      const pad = (a, b) => [a - 0.08 * (b - a || 10), b + 0.08 * (b - a || 10)];
      const xr = pad(Math.min(0, ...xs), Math.max(0, ...xs));
      const yr = pad(Math.min(0, ...ys), Math.max(0, ...ys));
      // The 1:1 line across the part of the chart where both axes overlap.
      const lo = Math.max(xr[0], yr[0]);
      const hi = Math.min(xr[1], yr[1]);
      const annotations = [];
      if (cmp) {
        const cp = pts.find((p) => p.m === cmp);
        if (cp) {
          annotations.push({
            x: cp.price_pct,
            y: cp.score_pct,
            text: "<b>" + esc(cmp.name) + "</b> (" + fmtPct(cp.price_pct) + " price, " + fmtPct(cp.score_pct) + " score)",
            showarrow: true,
            arrowhead: 2,
            arrowsize: 0.8,
            arrowcolor: c.muted,
            ax: 0,
            ay: -28,
            font: { size: 11, color: c.ink },
            bgcolor: c.surface,
            bordercolor: c.grid,
            borderpad: 3,
            borderwidth: 1,
          });
        }
      }
      const traces = [
        { type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: c.muted, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
        {
          type: "scatter",
          mode: "markers",
          x: xs,
          y: ys,
          customdata: pts.map((p) => p.m.model_key),
          text: pts.map(
            (p) =>
              "<b>" + esc(p.m.name) + "</b> vs " + esc(base.name) + "<br>Price: " + fmtPct(p.price_pct) + " (" + (p.price_diff_usd >= 0 ? "+" : "−") + fmtUsd(Math.abs(p.price_diff_usd)) + ")<br>Score: " + fmtPct(p.score_pct) + " (" + (p.score_diff >= 0 ? "+" : "−") + fmtInt(Math.abs(p.score_diff)) + " points)<br>Score per $: " + fmtPct(p.value_pct)
          ),
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: pts.map((p) => (p.m === cmp ? 16 : 10.5)),
            color: pts.map((p) => c.maker(p.m.maker)),
            line: { width: pts.map((p) => (p.m === cmp ? 3 : 1.5)), color: pts.map((p) => (p.m === cmp ? c.ink : c.surface)) },
          },
        },
        { type: "scatter", mode: "markers+text", x: [0], y: [0], text: ["Baseline: " + base.name], textposition: "top right", textfont: { color: c.ink, size: 11 }, marker: { size: 14, symbol: "diamond", color: c.ink }, hovertemplate: "Baseline: " + esc(base.name) + "<extra></extra>" },
      ];
      function drawComparison(target, xRange, yRange, closeUp) {
        const low = Math.max(xRange[0], yRange[0]);
        const high = Math.min(xRange[1], yRange[1]);
        const equality = Object.assign({}, traces[0], { x: [low, high], y: [low, high] });
        const baseline = Object.assign({}, traces[2], { mode: "markers" });
        const view = layout(c, {
          height: 360,
          margin: { t: 22, r: 16, b: 50, l: 55 },
          annotations: closeUp ? annotations.filter((p) => p.x >= xRange[0] && p.x <= xRange[1] && p.y >= yRange[0] && p.y <= yRange[1]) : [],
          xaxis: axis(c, "Price change (%)", { ticksuffix: "%", range: xRange, zeroline: true, zerolinecolor: c.muted }),
          yaxis: axis(c, "Blender score change (%)", { ticksuffix: "%", range: yRange, zeroline: true, zerolinecolor: c.muted }),
        });
        Plotly.react(target, [equality, traces[1], baseline], view, CONFIG);
        target.removeAllListeners("plotly_click");
        target.on("plotly_click", (event) => {
          const point = event.points && event.points[0];
          if (point && point.customdata) {
            S.compare[S.kind] = point.customdata;
            render();
          }
        });
      }
      drawComparison(el, [-50, 100], [-50, 100], true);
      drawComparison(ready("vd-premium-full"), xr, yr, false);
      const s = $("vd-premium-summary");
      if (!cmp) {
        s.innerHTML = "";
        return;
      }
      const u = upgrade(base, cmp);
      const verdict =
        u.value_pct > 0.5
          ? "improves score per dollar by <strong>+" + u.value_pct.toFixed(0) + "%</strong>"
          : u.value_pct < -0.5
          ? "reduces score per dollar by <strong>−" + Math.abs(u.value_pct).toFixed(0) + "%</strong>"
          : "leaves score per dollar about the same";
      s.innerHTML =
        "Moving from <strong>" + esc(base.name) + "</strong> (" + fmtUsd(base.price_usd) + ", " + fmtInt(base.score) + " pts) to <strong>" + esc(cmp.name) + "</strong> (" + fmtUsd(cmp.price_usd) + ", " + fmtInt(cmp.score) + " pts) " +
        (u.price_diff_usd >= 0 ? "costs <strong>" + fmtUsd(u.price_diff_usd) + " (" + fmtPct(u.price_pct) + ")</strong> more" : "saves <strong>" + fmtUsd(-u.price_diff_usd) + " (" + fmtPct(u.price_pct) + ")</strong>") +
        " and " + (u.score_diff >= 0 ? "adds " : "loses ") + "<strong>" + fmtInt(Math.abs(u.score_diff)) + " pts (" + fmtPct(u.score_pct) + ")</strong>. It " + verdict + ".";
    }

    // ------------------------------------------------------------ 4. value by budget
    function renderBudget(rows) {
      const el = $("vd-budget-out");
      const table = $("vd-budget-table");
      table.replaceChildren();
      const budget = Number(S.budget[S.kind]);
      const edges = S.edges[S.kind].split(/[,\s]+/).map(Number).filter((x) => x > 0 && isFinite(x));
      if (!rows.length) {
        el.innerHTML = '<div class="vd-state">No models match the filters.</div>';
        return;
      }
      let picks = "";
      if (budget > 0) {
        const within = rows.filter((m) => m.price_usd <= budget).sort((a, b) => b.score_per_dollar - a.score_per_dollar);
        const fastest = within.slice().sort((a, b) => b.score - a.score)[0];
        if (within.length) {
          const bv = within[0];
          const runnerUp = within[1] ? " · Runner-up: " + esc(within[1].name) + " (" + within[1].score_per_dollar.toFixed(2) + " pts/$)" : "";
          picks =
            '<div class="vd-cards vd-budget-cards">' +
              '<div class="vd-card vd-card-ok">' +
                '<div class="vd-card-title">Best Value ≤ ' + fmtUsd(budget) + " (" + within.length + " of " + rows.length + " " + KSHORT[S.kind] + "s)</div>" +
                '<div class="vd-big">' + esc(bv.name) + " <span>" + fmtUsd2(bv.price_usd) + "</span></div>" +
                '<div class="vd-small"><strong>' + bv.score_per_dollar.toFixed(2) + " pts/$</strong> · " + fmtInt(bv.score) + " pts" + runnerUp + "</div>" +
              "</div>" +
              '<div class="vd-card vd-card-gpu">' +
                '<div class="vd-card-title">Fastest Performance ≤ ' + fmtUsd(budget) + "</div>" +
                '<div class="vd-big">' + esc(fastest.name) + " <span>" + fmtUsd2(fastest.price_usd) + "</span></div>" +
                '<div class="vd-small"><strong>' + fmtInt(fastest.score) + " pts</strong> · " + fastest.score_per_dollar.toFixed(2) + " pts/$" +
                (fastest !== bv ? " · Extra spend buys peak speed at a lower pts/$ rate" : " · Also the #1 value pick in budget") + "</div>" +
              "</div>" +
            "</div>";
        } else {
          picks = '<p class="vd-note">No ' + KSHORT[S.kind] + " in view costs " + fmtUsd(budget) + " or less.</p>";
        }
      }
      const br = brackets(rows, edges);
      const maxRatio = Math.max(...br.map((b) => b.best_value_ratio || 0), 1);
      const label = (b) => (b.low === 0 ? "Under " + fmtUsd(b.high) : b.high === Infinity ? fmtUsd(b.low) + " and up" : fmtUsd(b.low) + " to " + fmtUsd(b.high));
      el.innerHTML = picks;
      table.innerHTML =
        '<div class="vd-table-wrap"><table class="table table-sm vd-table"><thead><tr><th>Price bracket</th><th class="num">Models</th><th>Best score per $</th><th>Highest score</th><th class="num">Median score</th><th class="num">Median price</th></tr></thead><tbody>' +
        br
          .map((b) =>
            b.n
              ? "<tr><td><strong>" + label(b) + '</strong></td><td class="num">' + b.n + "</td><td><div>" + esc(displayName(b.best_value)) + ' <span class="vd-badge">' + b.best_value_ratio.toFixed(2) + ' pts/$</span></div><div class="vd-mini-bar"><span style="width:' + Math.round((100 * b.best_value_ratio) / maxRatio) + '%"></span></div></td><td>' + esc(displayName(b.top_score_model)) + ' <span class="vd-muted">(' + fmtInt(b.top_score) + ' pts)</span></td><td class="num">' + fmtInt(b.median_score) + '</td><td class="num">' + fmtUsd(b.median_price) + "</td></tr>"
              : "<tr><td>" + label(b) + '</td><td class="num">0</td><td colspan="4" class="vd-muted">No ranked model in this bracket</td></tr>'
          )
          .join("") +
        "</tbody></table></div>";
    }

    // ------------------------------------------------------------ 5. distribution
    function renderDist(rows, c) {
      const gate = D.gates.percentiles;
      const n = rows.length;
      $("vd-dist-n").textContent = "Universe: ranked " + KSHORT[S.kind] + "s in view (models with a fresh price and a benchmark score). n = " + n + ".";
      if (n < gate) return emptyState("vd-dist", "Percentiles need at least " + gate + " models in view; " + n + " are. Widen the filters to see the distribution.");
      const el = ready("vd-dist");
      const sel = selectedModel(rows);
      const scores = rows.map((m) => m.score);
      const q25 = quantile(scores, 0.25);
      const q75 = quantile(scores, 0.75);
      const qs = [
        [0.25, "25th"],
        [0.5, "median"],
        [0.75, "75th"],
        [0.9, "90th"],
      ].map(([q, t]) => [quantile(scores, q), t]);
      const shapes = [
        { type: "rect", xref: "x", yref: "paper", x0: q25, x1: q75, y0: 0.08, y1: 0.92, fillcolor: c.quad, line: { width: 0 }, layer: "below" },
        ...qs.map(([v]) => ({ type: "line", xref: "x", yref: "paper", x0: v, x1: v, y0: 0, y1: 1, line: { color: c.muted, width: 1.2, dash: "dot" } })),
      ];
      const annotations = qs.map(([v, t]) => ({ x: v, y: 1, xref: "x", yref: "paper", yanchor: "bottom", showarrow: false, text: t + "<br>" + fmtInt(v), font: { size: 10, color: c.muted } }));
      // Deterministic jitter so dots with close scores do not sit on top of each other.
      const jitter = rows.map((m, i) => ((i * 37) % 11) / 11 - 0.5);
      if (sel) {
        const selIdx = rows.indexOf(sel);
        annotations.push({
          x: sel.score,
          y: jitter[selIdx],
          text: "<b>" + esc(sel.name) + "</b>",
          showarrow: true,
          arrowhead: 2,
          arrowsize: 0.8,
          arrowcolor: c.ink,
          ax: 0,
          ay: -28,
          font: { size: 11, color: c.ink },
          bgcolor: c.surface,
          bordercolor: c.grid,
          borderpad: 3,
          borderwidth: 1,
        });
      }
      Plotly.react(
        el,
        [
          {
            type: "scatter",
            mode: "markers",
            x: scores,
            y: jitter,
            customdata: rows.map((m) => m.model_key),
            text: rows.map((m) => "<b>" + esc(m.name) + "</b><br>Score: " + fmtInt(m.score) + " pts<br>Price: " + fmtUsd2(m.price_usd) + "<br>Faster than " + Math.round(100 * shareBelow(scores, m.score)) + "% of the other " + (n - 1) + " in view"),
            hovertemplate: "%{text}<extra></extra>",
            marker: {
              size: rows.map((m) => (m === sel ? 16 : 10.5)),
              color: rows.map((m) => c.maker(m.maker)),
              opacity: rows.map((m) => (sel && m !== sel ? 0.55 : 0.95)),
              line: { width: rows.map((m) => (m === sel ? 3 : 1.2)), color: rows.map((m) => (m === sel ? c.ink : c.surface)) },
            },
          },
        ],
        layout(c, {
          height: 240,
          margin: { t: 42, r: 16, b: 48, l: 16 },
          shapes,
          annotations,
          xaxis: axis(c, "Blender benchmark score (points)", { separatethousands: true }),
          yaxis: { visible: false, range: [-1, 1], fixedrange: true },
        }),
        CONFIG
      );
      bindClick(el);
      const s = $("vd-dist-summary");
      if (sel) {
        const sb = shareBelow(scores, sel.score);
        const tier = sb >= 2 / 3 ? "upper third" : sb >= 1 / 3 ? "middle third" : "lower third";
        s.innerHTML = "<strong>" + esc(sel.name) + "</strong> scores " + fmtInt(sel.score) + ": faster than " + Math.round(100 * sb) + "% of the other " + (n - 1) + " ranked " + KSHORT[S.kind] + "s in view, in the " + tier + " of this set. This is a position among the models this site can price, not among all hardware.";
      } else s.innerHTML = "";
    }

    // ------------------------------------------------------------ 6. frontier list
    function renderFrontier(rows) {
      const el = $("vd-frontier-out");
      const table = $("vd-frontier-table");
      table.replaceChildren();
      if (!rows.length) {
        el.innerHTML = '<div class="vd-state">No models match the filters.</div>';
        return;
      }
      const onF = frontier(rows);
      const eff = rows.filter((m, i) => onF[i]).sort((a, b) => a.price_usd - b.price_usd);
      const dom = rows
        .filter((m, i) => !onF[i])
        .map((m) => Object.assign({ m }, bestDominator(rows, m.model_key)))
        .sort((a, b) => a.m.price_usd - b.m.price_usd);
      const pills = eff
        .map((m) => '<span class="vd-badge vd-badge-ok">' + esc(m.name) + " · " + fmtUsd(m.price_usd) + "</span>")
        .join(" ");
      el.innerHTML =
        '<p class="vd-note"><strong>' + eff.length + " of " + rows.length + "</strong> " + KSHORT[S.kind] + "s in view are on the efficiency frontier.</p>";
      table.innerHTML =
        '<div class="vd-pill-row">' + pills + "</div>" +
        (dom.length
          ? '<div class="vd-table-wrap"><table class="table table-sm vd-table"><thead><tr><th>Dominated model</th><th class="num">Price</th><th class="num">Score</th><th>Better alternative</th><th class="num">More performance</th><th class="num">Lower cost</th></tr></thead><tbody>' +
            dom
              .map((d) => "<tr><td>" + esc(d.m.name) + (d.m.confidence === "low" ? ' <span class="vd-muted">(single listing)</span>' : "") + '</td><td class="num">' + fmtUsd(d.m.price_usd) + '</td><td class="num">' + fmtInt(d.m.score) + "</td><td><strong>" + esc(displayName(d.model_key)) + '</strong></td><td class="num">+' + d.more_score_pct.toFixed(0) + '%</td><td class="num">−' + d.lower_price_pct.toFixed(0) + "%</td></tr>")
              .join("") +
            "</tbody></table></div>"
          : '<p class="vd-note">No model in view is dominated.</p>');
    }

    // ------------------------------------------------------------ 7. price history
    function renderHistory(rows, c) {
      const keys = new Set(rows.map((m) => m.model_key));
      const hist = D.history.filter((h) => h.kind === S.kind && keys.has(h.model_key));
      const hs = D.history_summary;
      const meta = $("vd-hist-meta");
      const days = hs.days;
      meta.innerHTML = days.length
        ? "Collection started " + esc(days[0]) + "; last collection " + esc(days[days.length - 1]) + ". " + days.length + " collection day" + (days.length === 1 ? "" : "s") + ", " + (hs.gaps.length ? hs.gaps.length + " day" + (hs.gaps.length === 1 ? "" : "s") + " missed (" + hs.gaps.map(esc).join(", ") + ")" : "no days missed") + ". Points are observed prices only; lines join observations and are not estimates."
        : "";
      if (!hist.length) {
        $("vd-hist-summary").innerHTML = "";
        return emptyState("vd-hist", "No price history for the models in view.");
      }
      const el = ready("vd-hist");
      const sel = selectedModel(rows);
      const byKey = {};
      hist.forEach((h) => (byKey[h.model_key] = byKey[h.model_key] || []).push(h));
      // Keep the chart readable: show the efficiency frontier models + the selected model (or top 12 by value).
      const onF = frontier(rows);
      const focusKeys = new Set(rows.filter((m, i) => onF[i]).map((m) => m.model_key));
      rows
        .slice()
        .sort((a, b) => b.score_per_dollar - a.score_per_dollar)
        .slice(0, 10)
        .forEach((m) => focusKeys.add(m.model_key));
      if (sel) focusKeys.add(sel.model_key);
      const shownKeys = Object.keys(byKey).filter((k) => rows.length <= 16 || focusKeys.has(k));
      const allPrices = shownKeys.flatMap((k) => byKey[k].map((p) => p.price_usd));
      const minP = Math.min(...allPrices);
      const maxP = Math.max(...allPrices);
      const useLogY = maxP / Math.max(minP, 1) > 3.5;

      const traces = shownKeys.map((k) => {
        const m = models.find((x) => x.model_key === k && x.kind === S.kind);
        const pts = byKey[k].sort((a, b) => a.date.localeCompare(b.date));
        const isSel = sel && sel.model_key === k;
        return {
          type: "scatter",
          mode: "lines+markers",
          name: m.name,
          x: pts.map((p) => p.date),
          y: pts.map((p) => p.price_usd),
          customdata: pts.map(() => k),
          text: pts.map((p) => "<b>" + esc(m.name) + "</b><br>" + p.date + ": " + fmtUsd2(p.price_usd) + "<br>" + p.n_listings + " listing" + (p.n_listings === 1 ? "" : "s")),
          hovertemplate: "%{text}<extra></extra>",
          line: { color: c.maker(m.maker), width: isSel ? 3.2 : 1.5 },
          marker: { size: isSel ? 10 : 6.5, color: c.maker(m.maker), line: { width: isSel ? 2 : 0, color: c.ink } },
          opacity: sel && !isSel ? 0.38 : 1,
        };
      });
      const annotations = [];
      if (sel && byKey[sel.model_key]) {
        const pts = byKey[sel.model_key].slice().sort((a, b) => a.date.localeCompare(b.date));
        const lastPt = pts[pts.length - 1];
        annotations.push({
          x: lastPt.date,
          y: useLogY ? Math.log10(lastPt.price_usd) : lastPt.price_usd,
          text: "<b>" + esc(sel.name) + " · " + fmtUsd(lastPt.price_usd) + "</b>",
          showarrow: true,
          arrowhead: 2,
          arrowsize: 0.8,
          arrowcolor: c.ink,
          ax: -24,
          ay: -24,
          font: { size: 11, color: c.ink },
          bgcolor: c.surface,
          bordercolor: c.grid,
          borderpad: 3,
          borderwidth: 1,
        });
      }
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: 360,
          annotations,
          xaxis: axis(c, "Collection date", { type: "date", tickformat: "%-d %b", dtick: 86400000 }),
          yaxis: axis(
            c,
            "Lowest new listing (USD" + (useLogY ? ", log scale)" : ")"),
            useLogY
              ? { type: "log", tickprefix: "$", separatethousands: true, tickvals: LOG_TICKS, ticktext: LOG_TICKS.map((v) => "$" + v.toLocaleString("en-US")) }
              : { tickprefix: "$", separatethousands: true, rangemode: "tozero" }
          ),
        }),
        CONFIG
      );
      bindClick(el);
      const s = $("vd-hist-summary");
      const ms = hs.models.find((x) => sel && x.kind === S.kind && x.model_key === sel.model_key);
      if (!ms) {
        s.innerHTML = sel ? "No price history for " + esc(sel.name) + "." : "";
        return;
      }
      const base =
        "<strong>" + esc(sel.name) + "</strong>: " + ms.n_obs + " observation" + (ms.n_obs === 1 ? "" : "s") + " from " + esc(ms.first_date) + " to " + esc(ms.last_date) + ". Lowest " + fmtUsd2(ms.low) + ", median " + fmtUsd2(ms.median) + ", highest " + fmtUsd2(ms.high) + ". ";
      if (ms.trend_ok) {
        const pct = ms.change_pct;
        const word = Math.abs(pct) < 100 * D.gates.stable_share ? "stayed about the same" : pct > 0 ? "rose" : "fell";
        s.innerHTML = base + "Price " + word + ": " + (ms.change_usd >= 0 ? "+" : "−") + fmtUsd2(Math.abs(ms.change_usd)) + " (" + fmtPct(pct) + ") from first to last observation.";
      } else {
        s.innerHTML = base + '<span class="vd-muted">No trend is stated until history spans ' + D.gates.history_days + " days; it spans " + hs.span_days + ". A change over a day or two is noise.</span>";
      }
    }

    // ------------------------------------------------------------ wiring
    function render() {
      const rows = filtered();
      const c = colors();
      renderControls(rows);
      renderDetails(rows);
      renderScatter(rows, c);
      renderBars(rows, c);
      renderPremium(rows, c);
      renderBudget(rows);
      renderDist(rows, c);
      renderFrontier(rows);
      renderHistory(rows, c);
      if (root.TVPick) root.TVPick.render(S.kind, S.selected[S.kind]);
    }

    changeChartKind = (kind) => {
      S.kind = kind;
      render();
    };
    const num = (v) => (v === "" || v === null || isNaN(Number(v)) ? null : Number(v));
    $("vd-makers").addEventListener("change", (e) => {
      const mk = e.target.dataset.vdMaker;
      if (!mk) return;
      if (e.target.checked) S.makers[S.kind].add(mk);
      else S.makers[S.kind].delete(mk);
      render();
    });
    [["vd-pmin", "priceMin"], ["vd-pmax", "priceMax"], ["vd-smin", "scoreMin"], ["vd-smax", "scoreMax"]].forEach(([id, k]) =>
      $(id).addEventListener("input", (e) => {
        S[k] = num(e.target.value);
        render();
      })
    );
    $("vd-search").addEventListener("input", (e) => {
      S.search = e.target.value;
      render();
    });
    $("vd-hidelow").addEventListener("change", (e) => {
      S.hideLow = e.target.checked;
      render();
    });
    $("vd-reset").addEventListener("click", () => {
      S.makers[S.kind] = new Set(allOfKind().map((m) => m.maker));
      S.priceMin = S.priceMax = S.scoreMin = S.scoreMax = null;
      S.search = "";
      S.hideLow = false;
      ["vd-pmin", "vd-pmax", "vd-smin", "vd-smax", "vd-search"].forEach((id) => ($(id).value = ""));
      $("vd-hidelow").checked = false;
      render();
    });
    $("vd-select").addEventListener("change", (e) => {
      S.selected[S.kind] = e.target.value;
      if (root.TVPick) root.TVPick.set(S.kind, e.target.value);
      render();
    });
    $("vd-compare").addEventListener("change", (e) => {
      S.compare[S.kind] = e.target.value;
      render();
    });
    $("vd-medians").addEventListener("change", (e) => {
      S.medians = e.target.checked;
      render();
    });
    $("vd-logx").addEventListener("change", (e) => {
      S.logPrice = e.target.checked;
      render();
    });
    $("vd-frontier").addEventListener("change", (e) => {
      S.showFrontier = e.target.checked;
      render();
    });
    document.querySelectorAll("[data-vd-top]").forEach((b) =>
      b.addEventListener("click", () => {
        S.top = b.dataset.vdTop === "all" ? "all" : Number(b.dataset.vdTop);
        render();
      })
    );
    document.querySelectorAll("[data-vd-order]").forEach((b) =>
      b.addEventListener("click", () => {
        S.order = b.dataset.vdOrder;
        render();
      })
    );
    $("vd-budget").addEventListener("input", (e) => {
      S.budget[S.kind] = num(e.target.value);
      renderBudget(filtered());
    });
    $("vd-edges").addEventListener("change", (e) => {
      S.edges[S.kind] = e.target.value;
      renderBudget(filtered());
    });

    // Quarto's light/dark toggle swaps a class on <body>; redraw with that theme's colours.
    // The new stylesheet applies a moment after the class changes, so wait for it.
    let pending = null;
    new MutationObserver(() => {
      clearTimeout(pending);
      pending = setTimeout(render, 120);
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    render();
    // A shared link names its kind: open on that tab.
    const linked = root.TVPick && root.TVPick.urlKind();
    if (linked && linked !== S.kind) {
      const btn = document.querySelector('[data-vd-kind="' + linked + '"]');
      if (btn) btn.click();
    }
    // A saved dark preference can switch the stylesheet after this first draw.
    window.addEventListener("load", () => setTimeout(render, 120));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(typeof window !== "undefined" ? window : this);
