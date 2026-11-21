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
  const TITLES = { gpu: "Graphics card value", cpu: "Processor value" };
  // Colours follow the makers' own: NVIDIA green, AMD red-orange, Intel blue.
  const MAKER_SLOT = { Intel: 1, AMD: 2, NVIDIA: 3 };
  const LOG_TICKS = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const fmtDate = (iso) => {
    const p = String(iso || "").split("-");
    return p.length === 3 ? Number(p[2]) + " " + MONTHS[Number(p[1]) - 1] + " " + p[0] : "";
  };
  // "NVIDIA RTX 4060" -> "RTX 4060": the maker is shown by colour wherever the short name is used.
  const shortName = (m) => m.name.replace(/^(NVIDIA|AMD|Intel)\s+/, "");

  function boot() {
    const dataEl = $("vd-data");
    if (!dataEl) return;
    const D = JSON.parse(dataEl.textContent);
    const models = D.models.map((m) => Object.assign({}, m, { name: displayName(m.model_key) }));
    const hasPlotly = typeof Plotly !== "undefined";
    const S = {
      kind: "gpu",
      maker: "all",
      price: "any",
      scoreBand: "any",
      quality: "any",
      search: "",
      medians: true,
      showFrontier: true,
      logPrice: true,
      selected: {},
      compare: {},
      top: 10,
      closeUp: true,
      budget: {},
      edges: {},
    };
    for (const k of ["gpu", "cpu"]) {
      const part = models.filter((m) => m.kind === k);
      const normal = part.filter((m) => m.confidence !== "low");
      const best = bestValue(normal.length ? normal : part);
      S.selected[k] = best ? best.model_key : null;
      S.edges[k] = (D.brackets[k] || []).join(", ");
      S.budget[k] = part.length ? Math.round(quantile(part.map((m) => m.price_usd), 0.5)) : null;
      // A product picked on another Tech Value page (or in a shared link) wins over the default.
      const shared = root.TVPick && root.TVPick.get(k);
      if (shared && part.some((m) => m.model_key === shared)) S.selected[k] = shared;
    }

    function bestValue(rows) {
      return rows.reduce((b, m) => (!b || m.score_per_dollar > b.score_per_dollar ? m : b), null);
    }

    function colors() {
      const cs = getComputedStyle(document.body);
      const v = (n) => cs.getPropertyValue(n).trim();
      return {
        ink: v("--vd-ink"),
        muted: v("--vd-muted"),
        grid: v("--vd-grid"),
        surface: v("--vd-surface"),
        quad: v("--vx-quad"),
        ring: v("--vx-ring"),
        band: v("--vx-band"),
        maker: (m) => v("--vd-s" + (MAKER_SLOT[m] || 3)),
      };
    }

    function allOfKind() {
      return models.filter((m) => m.kind === S.kind);
    }

    // Price filter options: the budget brackets of this kind.
    function priceOptions() {
      const e = (D.brackets[S.kind] || []).slice().sort((a, b) => a - b);
      const out = [["any", "Any price"]];
      e.forEach((hi, i) => {
        const lo = i ? e[i - 1] : 0;
        out.push([lo + "-" + hi, lo ? fmtUsd(lo) + "–" + fmtUsd(hi) : "Under " + fmtUsd(hi)]);
      });
      if (e.length) out.push([e[e.length - 1] + "-", fmtUsd(e[e.length - 1]) + " and up"]);
      return out;
    }
    const SCORE_OPTIONS = [
      ["any", "Any score"],
      ["q75", "Top quarter"],
      ["q50", "Top half"],
      ["b50", "Bottom half"],
    ];
    const QUALITY_OPTIONS = [
      ["any", "Any quality"],
      ["multi", "Two or more listings"],
      ["single", "Single listing only"],
    ];

    function filtered() {
      const all = allOfKind();
      const q = S.search.trim().toLowerCase();
      const scores = all.map((m) => m.score);
      const q50 = quantile(scores, 0.5);
      const q75 = quantile(scores, 0.75);
      let pLo = 0;
      let pHi = Infinity;
      if (S.price !== "any") {
        const [a, b] = S.price.split("-");
        pLo = Number(a) || 0;
        pHi = b ? Number(b) : Infinity;
      }
      return all.filter(
        (m) =>
          (S.maker === "all" || m.maker === S.maker) &&
          m.price_usd >= pLo &&
          m.price_usd < pHi &&
          (S.scoreBand === "any" ||
            (S.scoreBand === "q75" && m.score >= q75) ||
            (S.scoreBand === "q50" && m.score >= q50) ||
            (S.scoreBand === "b50" && m.score < q50)) &&
          (S.quality === "any" || (S.quality === "single") === (m.confidence === "low")) &&
          (!q || m.model_key.includes(q) || m.name.toLowerCase().includes(q))
      );
    }

    function selectedModel(rows) {
      return rows.find((m) => m.model_key === S.selected[S.kind]) || null;
    }

    function select(key) {
      if (!key || !models.some((m) => m.model_key === key && m.kind === S.kind)) return;
      S.selected[S.kind] = key;
      if (root.TVPick) root.TVPick.set(S.kind, key);
      render();
    }

    function layout(c, extra) {
      return Object.assign(
        {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: c.ink, size: 12.5, family: getComputedStyle(document.body).fontFamily },
          margin: { t: 16, r: 16, b: 52, l: 64 },
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
          zeroline: false,
          linecolor: c.muted,
          linewidth: 1,
          ticks: "outside",
          tickcolor: c.grid,
          tickfont: { color: c.muted, size: 11 },
          automargin: true,
        },
        extra
      );
    }
    const CONFIG = { displayModeBar: false, responsive: true };

    function emptyState(id, text) {
      const el = $(id);
      if (hasPlotly) Plotly.purge(el);
      el.innerHTML = '<div class="vd-state">' + esc(text) + "</div>";
    }
    function ready(id) {
      const el = $(id);
      const st = el.querySelector(".vd-state");
      if (st) el.innerHTML = "";
      return el;
    }
    function bindClick(el, fn) {
      if (el.dataset.vdBound) return;
      el.dataset.vdBound = "1";
      el.on("plotly_click", (ev) => {
        const p = ev.points && ev.points[0];
        const key = p && (p.customdata || (p.data.customdata && p.data.customdata[p.pointIndex]));
        if (key) (fn || select)(key);
      });
    }
    // A ring drawn around the selected model, as in every chart on this page.
    function ring(c, x, y, size) {
      return {
        type: "scatter",
        mode: "markers",
        x: [x],
        y: [y],
        hoverinfo: "skip",
        marker: { size: size || 22, color: "rgba(0,0,0,0)", line: { width: 2.6, color: c.ring } },
      };
    }

    function hoverText(m) {
      const medLine = m.median_price_usd ? "Median listing: " + fmtUsd2(m.median_price_usd) + "<br>" : "";
      const stores = m.n_retailers ? " across " + m.n_retailers + " store" + (m.n_retailers === 1 ? "" : "s") : "";
      return (
        "<b>" + esc(m.name) + "</b><br>" +
        "Lowest new listing: <b>" + fmtUsd2(m.price_usd) + "</b><br>" +
        medLine +
        "Blender score: <b>" + fmtInt(m.score) + "</b> pts" + (m.backend ? " (" + esc(m.backend) + ")" : "") + "<br>" +
        "Score per $: <b>" + m.score_per_dollar.toFixed(2) + "</b><br>" +
        "Listings: " + (m.n_listings || 1) + stores
      );
    }

    // ------------------------------------------------------------ header and hero
    function renderHeader() {
      const all = allOfKind();
      document.querySelectorAll("[data-vd-kind]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.vdKind === S.kind))
      );
      $("vx-title").textContent = TITLES[S.kind];
      $("vx-checked").textContent = D.checked_date ? "Prices checked · " + fmtDate(D.checked_date) : "";
      document.querySelectorAll(".vx-kind").forEach((el) => (el.textContent = KSHORT[S.kind]));
      const normal = all.filter((m) => m.confidence !== "low");
      const best = bestValue(normal.length ? normal : all);
      const f = (D.funnel || {})[S.kind] || {};
      if (best) {
        $("vx-best-name").textContent = best.name;
        $("vx-best-meta").innerHTML =
          esc(best.score_per_dollar.toFixed(2)) + " pts/$ <span aria-hidden=\"true\">·</span> " + esc(fmtUsd(best.price_usd));
        $("vx-best").dataset.key = best.model_key;
      } else {
        $("vx-best-name").textContent = "No ranked " + KSHORT[S.kind] + "s yet";
        $("vx-best-meta").textContent = "";
      }
      const prices = all.map((m) => m.price_usd);
      $("vx-range").textContent = all.length >= 4 ? fmtUsd(quantile(prices, 0.25)) + "–" + fmtUsd(quantile(prices, 0.75)) : "Too few models";
      $("vx-coverage").textContent = (f.ranked ?? all.length) + " of " + (f.price_checked ?? all.length);
    }

    // ------------------------------------------------------------ filters
    function fillSelect(id, options, value) {
      const el = $(id);
      el.innerHTML = options
        .map(([v, t]) => '<option value="' + esc(v) + '"' + (v === value ? " selected" : "") + ">" + esc(t) + "</option>")
        .join("");
    }
    function renderFilters(rows) {
      const all = allOfKind();
      const makers = [...new Set(all.map((m) => m.maker))].sort();
      if (S.maker !== "all" && !makers.includes(S.maker)) S.maker = "all";
      fillSelect("vd-maker", [["all", "All makers"], ...makers.map((m) => [m, m])], S.maker);
      const po = priceOptions();
      if (!po.some(([v]) => v === S.price)) S.price = "any";
      fillSelect("vd-price", po, S.price);
      fillSelect("vd-score", SCORE_OPTIONS, S.scoreBand);
      fillSelect("vd-quality", QUALITY_OPTIONS, S.quality);
      $("vd-search").placeholder = S.kind === "gpu" ? "Search model (e.g., RTX 4060)" : "Search model (e.g., Ryzen 7)";
      const filteredNow = rows.length !== all.length;
      $("vd-count").textContent = filteredNow ? "Showing " + rows.length + " of " + all.length + " ranked " + KSHORT[S.kind] + "s" : "";
      $("vd-reset").hidden = !filteredNow;
      $("vd-medians").checked = S.medians;
      $("vd-frontier").checked = S.showFrontier;
      $("vd-logx").checked = S.logPrice;
      $("vd-top").value = String(S.top);
      document.querySelectorAll("[data-vd-zoom]").forEach((b) =>
        b.setAttribute("aria-pressed", String((b.dataset.vdZoom === "close") === S.closeUp))
      );
      $("vd-budget").value = S.budget[S.kind] ?? "";
      $("vd-edges").value = S.edges[S.kind];
    }

    // ------------------------------------------------------------ selected model inspector
    function renderInspector(rows) {
      const m = selectedModel(rows);
      const sel = $("vd-select");
      const sorted = rows.slice().sort((a, b) => b.score_per_dollar - a.score_per_dollar);
      sel.innerHTML = sorted.length
        ? sorted.map((r) => '<option value="' + esc(r.model_key) + '"' + (m && r === m ? " selected" : "") + ">" + esc(shortName(r)) + "</option>").join("")
        : "<option>No match</option>";
      if (!m) {
        sel.insertAdjacentHTML("afterbegin", '<option selected disabled value="">Choose a model</option>');
      }
      $("vx-insp").classList.toggle("vx-insp-empty", !m);
      sel.classList.toggle("is-long", !!m && shortName(m).length > 11);
      $("vx-insp-name-bar").className = "vx-insp-bar vd-s" + (m ? MAKER_SLOT[m.maker] || 3 : 3);
      if (!m) {
        $("vx-insp-score").textContent = "";
        $("vx-insp-runs").textContent = "The filters hide the selected model. Pick one above or in a chart.";
        $("vx-insp-price").textContent = "";
        $("vx-insp-listings").textContent = "";
        return;
      }
      const rk = sorted.indexOf(m) + 1;
      $("vx-insp-score").textContent = fmtInt(m.score) + " Blender pts";
      $("vx-insp-runs").textContent = fmtInt(m.n_runs) + " benchmark runs";
      $("vx-insp-price").innerHTML =
        "<strong>" + esc(fmtUsd2(m.price_usd)) + "</strong> · " + esc(m.score_per_dollar.toFixed(2)) + " pts/$ · #" + rk + " of " + rows.length;
      const n = m.n_listings || 1;
      const st = m.n_retailers || 1;
      $("vx-insp-listings").textContent =
        n + " listing" + (n === 1 ? "" : "s") + " · " + st + " store" + (st === 1 ? "" : "s");
      const setLinks = () => {
        if (!root.TVPick) return;
        $("vx-link-fair").href = root.TVPick.href("fair-price.html", S.kind, m.model_key);
        $("vx-link-perf").href = root.TVPick.href("residuals.html", S.kind, m.model_key);
      };
      setLinks();
      if (root.TVPick) root.TVPick.load().then(setLinks);
    }

    // ------------------------------------------------------------ performance vs price
    function renderScatter(rows, c) {
      if (!rows.length) {
        $("vd-scatter-legend").replaceChildren();
        return emptyState("vd-scatter", "No models match the filters.");
      }
      const el = ready("vd-scatter");
      const onF = frontier(rows);
      const sel = selectedModel(rows);
      const makers = [...new Set(rows.map((m) => m.maker))].sort();
      const xs = rows.map((m) => m.price_usd);
      const ys = rows.map((m) => m.score);
      const medX = median(xs);
      const medY = median(ys);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      const x0 = S.logPrice ? minX / 1.3 : 0;
      const x1 = S.logPrice ? maxX * 1.2 : maxX * 1.05;
      const y1 = maxY * 1.08;
      const traces = [];
      const shapes = [];
      const annotations = [];
      const lx = (v) => (S.logPrice ? Math.log10(v) : v);
      if (S.medians && rows.length >= 2) {
        shapes.push(
          { type: "rect", xref: "x", yref: "y", x0, x1: medX, y0: medY, y1, fillcolor: c.quad, line: { width: 0 }, layer: "below" },
          { type: "line", xref: "x", yref: "y", x0: medX, x1: medX, y0: 0, y1, line: { color: c.muted, width: 1, dash: "dot" } },
          { type: "line", xref: "x", yref: "y", x0, x1, y0: medY, y1: medY, line: { color: c.muted, width: 1, dash: "dot" } }
        );
        annotations.push(
          { xref: "x", yref: "y", x: lx(x0), y: y1, xanchor: "left", yanchor: "top", xshift: 12, yshift: -10, align: "left", showarrow: false, text: "<i><b>Faster and cheaper<br>than median</b></i>", font: { size: 13, color: c.ring } },
          { xref: "x", yref: "y", x: lx(medX), y: y1, xanchor: "left", yanchor: "top", xshift: 6, showarrow: false, text: "Median " + fmtUsd(medX), font: { size: 11, color: c.muted } },
          { xref: "x", yref: "y", x: lx(x1), y: medY, xanchor: "right", yanchor: "bottom", showarrow: false, text: "Median " + fmtInt(medY) + " pts", font: { size: 11, color: c.muted } }
        );
      }
      if (S.showFrontier) {
        const eff = rows.filter((m, i) => onF[i]).sort((a, b) => a.price_usd - b.price_usd);
        traces.push({
          type: "scatter",
          mode: "lines",
          x: [...eff.map((m) => m.price_usd), x1],
          y: [...eff.map((m) => m.score), eff.length ? eff[eff.length - 1].score : 0],
          line: { shape: "hv", color: c.muted, width: 1.3, dash: "dash" },
          hoverinfo: "skip",
        });
      }
      makers.forEach((mk) => {
        const part = rows.filter((m) => m.maker === mk);
        traces.push({
          type: "scatter",
          mode: "markers",
          name: mk,
          x: part.map((m) => m.price_usd),
          y: part.map((m) => m.score),
          customdata: part.map((m) => m.model_key),
          text: part.map(hoverText),
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: 9,
            color: part.map((m) => (m.confidence === "low" ? "rgba(0,0,0,0)" : c.maker(mk))),
            line: { width: part.map((m) => (m.confidence === "low" ? 1.8 : 0)), color: c.maker(mk) },
          },
        });
      });
      if (sel) traces.push(ring(c, sel.price_usd, sel.score));
      // Ticks inside the range only; a narrow chart keeps one tick per doubling or so.
      const ticks = (el.clientWidth < 520 ? [100, 200, 500, 1000, 2000, 5000, 10000] : LOG_TICKS).filter((v) => v >= x0 && v <= x1);
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: el.clientWidth < 520 ? 340 : 400,
          margin: { t: 12, r: 14, b: 50, l: 64 },
          shapes,
          annotations,
          xaxis: axis(c, "Lowest new listing (USD" + (S.logPrice ? ", log scale)" : ")"), S.logPrice
            ? { type: "log", tickvals: ticks, ticktext: ticks.map((v) => "$" + v.toLocaleString("en-US")), tickangle: 0, range: [Math.log10(x0), Math.log10(x1)] }
            : { tickprefix: "$", separatethousands: true, range: [0, x1] }),
          yaxis: axis(c, "Blender score (points)", { separatethousands: true, range: [0, y1] }),
        }),
        CONFIG
      );
      bindClick(el);
      $("vd-scatter-legend").innerHTML =
        makers.map((mk) => '<span><span class="vd-swatch vx-dot vd-s' + (MAKER_SLOT[mk] || 3) + '"></span>' + esc(mk) + "</span>").join("") +
        (rows.some((m) => m.confidence === "low") ? '<span><span class="vd-swatch vx-dot vx-hollow"></span>Single listing</span>' : "");
    }

    // ------------------------------------------------------------ value leaderboard
    function renderBars(rows) {
      const el = $("vd-bars");
      if (!rows.length) {
        el.innerHTML = '<div class="vd-state">No models match the filters.</div>';
        return;
      }
      let sorted = rows.slice().sort((a, b) => b.score_per_dollar - a.score_per_dollar);
      if (S.top !== "all") sorted = sorted.slice(0, S.top);
      const max = sorted[0].score_per_dollar;
      const sel = selectedModel(rows);
      el.innerHTML = sorted
        .map(
          (m) =>
            '<button type="button" class="vx-bar-row' + (m === sel ? " is-selected" : "") + (m.confidence === "low" ? " is-single" : "") +
            '" data-key="' + esc(m.model_key) + '" title="' + esc(m.name + " · " + fmtUsd2(m.price_usd) + " · " + fmtInt(m.score) + " pts" + (m.confidence === "low" ? " · single listing" : "")) + '">' +
            '<span class="vx-bar-name">' + esc(shortName(m)) + "</span>" +
            '<span class="vx-bar-track"><span class="vd-s' + (MAKER_SLOT[m.maker] || 3) + '" style="width:' + ((100 * m.score_per_dollar) / max).toFixed(1) + '%"></span></span>' +
            '<span class="vx-bar-val">' + m.score_per_dollar.toFixed(2) + "</span></button>"
        )
        .join("");
    }

    // ------------------------------------------------------------ price premium vs performance gain
    function renderPremium(rows, c) {
      const base = selectedModel(rows);
      const s = $("vd-premium-summary");
      if (!base || rows.length < 2) {
        s.innerHTML = "";
        return emptyState("vd-premium", base ? "Only one model is in view; there is nothing to compare it with." : "Select a model to compare.");
      }
      const others = rows.filter((m) => m !== base);
      if (!others.some((m) => m.model_key === S.compare[S.kind])) {
        // Default comparison: the next model up on the efficiency frontier, the upgrade worth weighing.
        const onF = frontier(rows);
        const up = (set) => set.filter((m) => m.price_usd > base.price_usd).sort((a, b) => a.price_usd - b.price_usd)[0];
        const pick = up(rows.filter((m, i) => onF[i] && m !== base)) || up(others) || others[0];
        S.compare[S.kind] = pick.model_key;
      }
      const cmp = others.find((m) => m.model_key === S.compare[S.kind]);
      const el = ready("vd-premium");
      const pts = others.map((m) => Object.assign({ m }, upgrade(base, m)));
      const xs = pts.map((p) => p.price_pct);
      const ys = pts.map((p) => p.score_pct);
      const pad = (a, b) => [a - 0.06 * (b - a || 10), b + 0.06 * (b - a || 10)];
      const xr = S.closeUp ? [-50, 100] : pad(Math.min(0, ...xs), Math.max(0, ...xs));
      const yr = S.closeUp ? [-50, 100] : pad(Math.min(0, ...ys), Math.max(0, ...ys));
      const lo = Math.max(xr[0], yr[0]);
      const hi = Math.min(xr[1], yr[1]);
      const cp = cmp && pts.find((p) => p.m === cmp);
      const traces = [
        { type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: c.muted, width: 1.2, dash: "dash" }, hoverinfo: "skip" },
        {
          type: "scatter",
          mode: "markers",
          x: xs,
          y: ys,
          customdata: pts.map((p) => p.m.model_key),
          text: pts.map(
            (p) =>
              "<b>" + esc(p.m.name) + "</b> vs " + esc(base.name) + "<br>Price: " + fmtPct(p.price_pct) + " (" + (p.price_diff_usd >= 0 ? "+" : "−") + fmtUsd(Math.abs(p.price_diff_usd)) + ")<br>Score: " + fmtPct(p.score_pct) + "<br>Score per $: " + fmtPct(p.value_pct)
          ),
          hovertemplate: "%{text}<extra></extra>",
          marker: { size: 8, color: pts.map((p) => c.maker(p.m.maker)) },
        },
        { type: "scatter", mode: "markers", x: [0], y: [0], marker: { size: 11, symbol: "diamond", color: c.ink }, hovertemplate: "Selected: " + esc(base.name) + "<extra></extra>" },
      ];
      if (cp) traces.push(ring(c, cp.price_pct, cp.score_pct, 18));
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: 260,
          margin: { t: 8, r: 10, b: 46, l: 56 },
          xaxis: axis(c, "Price change (%)", { ticksuffix: "%", range: xr }),
          yaxis: axis(c, "Blender score change (%)", { ticksuffix: "%", range: yr }),
        }),
        CONFIG
      );
      bindClick(el, (key) => {
        S.compare[S.kind] = key;
        render();
      });
      if (!cp) {
        s.innerHTML = "";
        return;
      }
      s.innerHTML =
        "<strong>" + esc(shortName(base)) + " → " + esc(shortName(cmp)) + "</strong><span aria-hidden=\"true\"> · </span>" +
        "<span>" + fmtPct(cp.price_pct) + " price</span><span aria-hidden=\"true\"> · </span>" +
        "<strong>" + fmtPct(cp.score_pct) + " score</strong>";
      s.title = "Score per dollar " + fmtPct(cp.value_pct) + ". Click another dot to compare with it.";
    }

    // ------------------------------------------------------------ value by budget
    function renderBudget(rows) {
      const el = $("vd-budget-out");
      const budget = Number(S.budget[S.kind]);
      const within = budget > 0 ? rows.filter((m) => m.price_usd <= budget) : [];
      if (!rows.length) {
        el.innerHTML = '<div class="vd-state">No models match the filters.</div>';
      } else if (!(budget > 0)) {
        el.innerHTML = '<p class="vx-muted">Enter a budget in US dollars.</p>';
      } else if (!within.length) {
        el.innerHTML = '<p class="vx-muted">No ' + KSHORT[S.kind] + " in view costs " + fmtUsd(budget) + " or less.</p>";
      } else {
        const bv = bestValue(within);
        const fast = within.reduce((b, m) => (m.score > b.score ? m : b));
        const row = (label, m, detail) =>
          '<button type="button" class="vx-budget-row" data-key="' + esc(m.model_key) + '" title="' + esc(detail) + '"><span>' + label + "</span><strong>" + esc(shortName(m)) + "</strong></button>";
        el.innerHTML =
          row("Best value", bv, fmtUsd2(bv.price_usd) + " · " + bv.score_per_dollar.toFixed(2) + " pts/$") +
          row("Fastest within budget", fast, fmtUsd2(fast.price_usd) + " · " + fmtInt(fast.score) + " pts");
      }
      renderBracketTable(rows);
    }

    function renderBracketTable(rows) {
      const table = $("vd-budget-table");
      const edges = S.edges[S.kind].split(/[,\s]+/).map(Number).filter((x) => x > 0 && isFinite(x));
      if (!rows.length) {
        table.innerHTML = "";
        return;
      }
      const br = brackets(rows, edges);
      const label = (b) => (b.low === 0 ? "Under " + fmtUsd(b.high) : b.high === Infinity ? fmtUsd(b.low) + " and up" : fmtUsd(b.low) + " to " + fmtUsd(b.high));
      table.innerHTML =
        '<div class="vd-table-wrap"><table class="table table-sm vd-table"><thead><tr><th>Price bracket</th><th class="num">Models</th><th>Best score per $</th><th>Highest score</th><th class="num">Median score</th><th class="num">Median price</th></tr></thead><tbody>' +
        br
          .map((b) =>
            b.n
              ? "<tr><td><strong>" + label(b) + '</strong></td><td class="num">' + b.n + "</td><td>" + esc(displayName(b.best_value)) + ' <span class="vd-muted">(' + b.best_value_ratio.toFixed(2) + " pts/$)</span></td><td>" + esc(displayName(b.top_score_model)) + ' <span class="vd-muted">(' + fmtInt(b.top_score) + ' pts)</span></td><td class="num">' + fmtInt(b.median_score) + '</td><td class="num">' + fmtUsd(b.median_price) + "</td></tr>"
              : "<tr><td>" + label(b) + '</td><td class="num">0</td><td colspan="4" class="vd-muted">No ranked model in this bracket</td></tr>'
          )
          .join("") +
        "</tbody></table></div>";
    }

    // ------------------------------------------------------------ performance distribution
    function renderDist(rows, c) {
      const gate = D.gates.percentiles;
      const n = rows.length;
      if (n < gate) {
        $("vd-dist-summary").innerHTML = "";
        return emptyState("vd-dist", "Percentiles need at least " + gate + " models in view; " + n + " are.");
      }
      const el = ready("vd-dist");
      const sel = selectedModel(rows);
      const scores = rows.map((m) => m.score);
      const qs = [0.25, 0.5, 0.75, 0.9].map((q) => [quantile(scores, q), "P" + Math.round(q * 100)]);
      const shapes = [
        { type: "rect", xref: "x", yref: "paper", x0: qs[0][0], x1: qs[2][0], y0: 0.12, y1: 0.78, fillcolor: c.band, line: { width: 0 }, layer: "below" },
        ...qs.map(([v]) => ({ type: "line", xref: "x", yref: "paper", x0: v, x1: v, y0: 0, y1: 0.86, line: { color: c.muted, width: 1, dash: "dash" } })),
      ];
      const annotations = qs.map(([v, t]) => ({
        x: v, y: 0.88, xref: "x", yref: "paper", yanchor: "bottom", showarrow: false,
        text: t, hovertext: t + ": " + fmtInt(v) + " pts", font: { size: 11, color: c.ink },
      }));
      const jitter = rows.map((m, i) => (((i * 37) % 11) / 11 - 0.5) * 0.5);
      const traces = [
        {
          type: "scatter",
          mode: "markers",
          x: scores,
          y: jitter,
          customdata: rows.map((m) => m.model_key),
          text: rows.map((m) => "<b>" + esc(m.name) + "</b><br>" + fmtInt(m.score) + " pts · " + fmtUsd(m.price_usd) + "<br>Faster than " + Math.round(100 * shareBelow(scores, m.score)) + "% of the others in view"),
          hovertemplate: "%{text}<extra></extra>",
          marker: { size: 8, color: rows.map((m) => c.maker(m.maker)) },
        },
      ];
      if (sel) traces.push(ring(c, sel.score, jitter[rows.indexOf(sel)], 18));
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: 150,
          margin: { t: 8, r: 14, b: 34, l: 14 },
          shapes,
          annotations,
          xaxis: axis(c, "", { separatethousands: true, showline: false, ticks: "", tickfont: { color: c.muted, size: 10.5 } }),
          yaxis: { visible: false, range: [-0.8, 1.15], fixedrange: true },
        }),
        CONFIG
      );
      bindClick(el);
      const s = $("vd-dist-summary");
      if (sel) {
        const sb = shareBelow(scores, sel.score);
        s.innerHTML = "<strong>" + esc(shortName(sel)) + "</strong> is faster than " + Math.round(100 * sb) + "% of the other " + (n - 1) + " " + KSHORT[S.kind] + "s in view.";
      } else s.innerHTML = "";
    }

    // ------------------------------------------------------------ efficiency frontier models
    function renderFrontier(rows) {
      const el = $("vd-frontier-out");
      const table = $("vd-frontier-table");
      if (!rows.length) {
        el.innerHTML = '<div class="vd-state">No models match the filters.</div>';
        table.replaceChildren();
        return;
      }
      const onF = frontier(rows);
      const eff = rows.filter((m, i) => onF[i]).sort((a, b) => a.price_usd - b.price_usd);
      const dom = rows
        .filter((m, i) => !onF[i])
        .map((m) => Object.assign({ m }, bestDominator(rows, m.model_key)))
        .sort((a, b) => a.m.price_usd - b.m.price_usd);
      el.innerHTML =
        '<p class="vx-muted"><strong>' + eff.length + " of " + rows.length + "</strong> " + KSHORT[S.kind] + "s in view are on the efficiency frontier: no other model is both cheaper and faster. Each sets a new high score as the budget rises.</p>" +
        '<div class="vd-pill-row">' +
        eff.map((m) => '<button type="button" class="vx-pill" data-key="' + esc(m.model_key) + '"><span class="vd-swatch vx-dot vd-s' + (MAKER_SLOT[m.maker] || 3) + '"></span>' + esc(shortName(m)) + " · " + fmtUsd(m.price_usd) + "</button>").join("") +
        "</div>";
      table.innerHTML = dom.length
        ? '<div class="vd-table-wrap"><table class="table table-sm vd-table"><thead><tr><th>Dominated model</th><th class="num">Price</th><th class="num">Score</th><th>Better alternative</th><th class="num">More performance</th><th class="num">Lower cost</th></tr></thead><tbody>' +
          dom
            .map((d) => "<tr><td>" + esc(d.m.name) + (d.m.confidence === "low" ? ' <span class="vd-muted">(single listing)</span>' : "") + '</td><td class="num">' + fmtUsd(d.m.price_usd) + '</td><td class="num">' + fmtInt(d.m.score) + "</td><td><strong>" + esc(displayName(d.model_key)) + '</strong></td><td class="num">+' + d.more_score_pct.toFixed(0) + '%</td><td class="num">−' + d.lower_price_pct.toFixed(0) + "%</td></tr>")
            .join("") +
          "</tbody></table></div>"
        : '<p class="vx-muted">No model in view is dominated.</p>';
    }

    // ------------------------------------------------------------ price history
    function renderHistory(rows, c) {
      const keys = new Set(rows.map((m) => m.model_key));
      const hist = D.history.filter((h) => h.kind === S.kind && keys.has(h.model_key));
      const hs = D.history_summary;
      const days = hs.days;
      $("vd-hist-meta").textContent = days.length
        ? "Lowest new listing each collection day, " + fmtDate(days[0]) + " to " + fmtDate(days[days.length - 1]) +
          (hs.gaps.length ? " (" + hs.gaps.length + " day" + (hs.gaps.length === 1 ? "" : "s") + " missed)" : "") + ". The selected model and the top value picks are shown."
        : "";
      if (!hist.length) {
        $("vd-hist-summary").innerHTML = "";
        return emptyState("vd-hist", "No price history for the models in view.");
      }
      const el = ready("vd-hist");
      const sel = selectedModel(rows);
      const byKey = {};
      hist.forEach((h) => (byKey[h.model_key] = byKey[h.model_key] || []).push(h));
      // The selected model plus the best value picks, three lines in all, as in the leaderboard.
      const order = [];
      if (sel && byKey[sel.model_key]) order.push(sel);
      rows
        .slice()
        .sort((a, b) => b.score_per_dollar - a.score_per_dollar)
        .forEach((m) => {
          if (order.length < 3 && byKey[m.model_key] && !order.includes(m)) order.push(m);
        });
      const palette = [c.maker("NVIDIA"), c.maker("Intel"), c.maker("AMD")];
      const useMaker = new Set(order.map((m) => m.maker)).size === order.length;
      const traces = order.map((m, i) => {
        const pts = byKey[m.model_key].slice().sort((a, b) => a.date.localeCompare(b.date));
        const color = useMaker ? c.maker(m.maker) : palette[i];
        const isSel = sel && m === sel;
        return {
          type: "scatter",
          mode: pts.length > 1 ? "lines" : "markers",
          name: shortName(m),
          x: pts.map((p) => p.date),
          y: pts.map((p) => p.price_usd),
          customdata: pts.map(() => m.model_key),
          text: pts.map((p) => "<b>" + esc(m.name) + "</b><br>" + fmtDate(p.date) + ": " + fmtUsd2(p.price_usd) + "<br>" + p.n_listings + " listing" + (p.n_listings === 1 ? "" : "s")),
          hovertemplate: "%{text}<extra></extra>",
          line: { color, width: isSel ? 2.6 : 1.8 },
          marker: { size: 7, color },
          fill: isSel ? "tozeroy" : "none",
          fillcolor: isSel ? c.band : undefined,
        };
      });
      const narrow = el.clientWidth < 560;
      Plotly.react(
        el,
        traces,
        layout(c, {
          height: 300,
          showlegend: true,
          legend: narrow
            ? { orientation: "h", x: 0, y: -0.28, font: { size: 12, color: c.ink } }
            : { x: 1.02, xanchor: "left", y: 1, font: { size: 12, color: c.ink } },
          margin: { t: 10, r: narrow ? 10 : 20, b: narrow ? 80 : 46, l: 64 },
          xaxis: axis(c, "Date", { type: "date", tickformat: "%-d %b", dtick: 86400000 }),
          yaxis: axis(c, "Price (USD)", { tickprefix: "$", separatethousands: true, rangemode: "tozero" }),
        }),
        CONFIG
      );
      bindClick(el);
      const s = $("vd-hist-summary");
      const ms = sel && hs.models.find((x) => x.kind === S.kind && x.model_key === sel.model_key);
      if (!ms) {
        s.innerHTML = sel ? "No price history for " + esc(sel.name) + "." : "";
        return;
      }
      let trend = '<span class="vx-muted">No trend is stated until the history spans ' + D.gates.history_days + " days; it spans " + hs.span_days + ".</span>";
      if (ms.trend_ok) {
        const pct = ms.change_pct;
        const word = Math.abs(pct) < 100 * D.gates.stable_share ? "stayed about the same" : pct > 0 ? "rose" : "fell";
        trend = "Price " + word + " (" + fmtPct(pct) + ") from first to last observation.";
      }
      s.innerHTML =
        "<strong>" + esc(shortName(sel)) + "</strong>: lowest " + fmtUsd2(ms.low) + ", median " + fmtUsd2(ms.median) + ", highest " + fmtUsd2(ms.high) + " over " + ms.n_obs + " observation" + (ms.n_obs === 1 ? "" : "s") + ". " + trend;
    }

    // ------------------------------------------------------------ wiring
    function render() {
      const rows = filtered();
      renderHeader();
      renderFilters(rows);
      renderInspector(rows);
      renderBars(rows);
      renderBudget(rows);
      renderFrontier(rows);
      if (!hasPlotly) return;
      const c = colors();
      renderScatter(rows, c);
      renderPremium(rows, c);
      renderDist(rows, c);
      renderHistory(rows, c);
    }

    if (!hasPlotly) {
      document.querySelectorAll(".vd-chart").forEach((el) => {
        el.innerHTML = '<div class="vd-state vd-error">The chart library did not load, so the charts cannot be drawn. The leaderboard and tables still work.</div>';
      });
    }

    document.querySelectorAll("[data-vd-kind]").forEach((b) =>
      b.addEventListener("click", () => {
        S.kind = b.dataset.vdKind;
        render();
      })
    );
    const num = (v) => (v === "" || v === null || isNaN(Number(v)) ? null : Number(v));
    [["vd-maker", "maker"], ["vd-price", "price"], ["vd-score", "scoreBand"], ["vd-quality", "quality"]].forEach(([id, k]) =>
      $(id).addEventListener("change", (e) => {
        S[k] = e.target.value;
        render();
      })
    );
    $("vd-search").addEventListener("input", (e) => {
      S.search = e.target.value;
      render();
    });
    $("vd-reset").addEventListener("click", () => {
      S.price = S.scoreBand = S.quality = "any";
      S.maker = "all";
      S.search = "";
      $("vd-search").value = "";
      render();
    });
    $("vd-select").addEventListener("change", (e) => select(e.target.value));
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
    $("vd-top").addEventListener("change", (e) => {
      S.top = e.target.value === "all" ? "all" : Number(e.target.value);
      render();
    });
    document.querySelectorAll("[data-vd-zoom]").forEach((b) =>
      b.addEventListener("click", () => {
        S.closeUp = b.dataset.vdZoom === "close";
        render();
      })
    );
    // Rows, pills and the hero card select a model.
    document.addEventListener("click", (e) => {
      const t = e.target.closest(".vx-bar-row, .vx-budget-row, .vx-pill, #vx-best");
      if (t && t.dataset.key) select(t.dataset.key);
    });
    $("vd-budget").addEventListener("input", (e) => {
      S.budget[S.kind] = num(e.target.value);
      renderBudget(filtered());
    });
    $("vd-edges").addEventListener("change", (e) => {
      S.edges[S.kind] = e.target.value;
      renderBracketTable(filtered());
    });

    // Quarto's light/dark toggle swaps a class on <body>; redraw with that theme's colours.
    // The new stylesheet applies a moment after the class changes, so wait for it.
    let pending = null;
    new MutationObserver(() => {
      clearTimeout(pending);
      pending = setTimeout(render, 120);
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    // A shared link names its kind: open on that tab.
    const linked = root.TVPick && root.TVPick.urlKind();
    if (linked === "cpu" || linked === "gpu") S.kind = linked;
    render();
    // A saved dark preference can switch the stylesheet after this first draw.
    window.addEventListener("load", () => setTimeout(render, 120));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(typeof window !== "undefined" ? window : this);
