// One selected product across the three Tech Value pages, plus a compact panel that shows what
// each page says about it.
//
// The pick lives in the URL (?pick=gpu:nvidia rtx 4060) so a link carries it, and in local
// storage so moving between pages keeps it. The panel reads the shared hardware catalogue
// (data/v1/tech-value/catalogue.json), which joins Value ranking, Fair price and the
// over- and under-performer page on one model key.
(function (root) {
  "use strict";
  const STORE = "pdi-tv-pick";
  const DATA_URL = "../data/v1/tech-value/catalogue.json";
  const PAGES = [
    ["index.html", "Value ranking"],
    ["fair-price.html", "Fair price"],
    ["residuals.html", "Over & under performers"],
  ];

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
  const pct = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(0) + "%";

  function fromUrl() {
    try {
      const p = new URLSearchParams(root.location.search).get("pick");
      const i = p ? p.indexOf(":") : -1;
      return i > 0 ? { kind: p.slice(0, i), key: p.slice(i + 1) } : null;
    } catch (e) {
      return null;
    }
  }

  function stored() {
    try {
      return JSON.parse(root.localStorage.getItem(STORE) || "{}") || {};
    } catch (e) {
      return {};
    }
  }

  // The product picked for this kind: the URL wins, then the last pick on any page.
  function get(kind) {
    const u = fromUrl();
    if (u && u.kind === kind) return u.key;
    return stored()[kind] || null;
  }

  // The kind named in the URL, if any, so a shared link opens on the right tab.
  function urlKind() {
    const u = fromUrl();
    return u ? u.kind : null;
  }

  function href(page, kind, key) {
    return page + "?pick=" + encodeURIComponent(kind + ":" + key) + (kind === "gpu" && page === "residuals.html" ? "#gpu" : "");
  }

  // Record a pick the reader made. Call only on a reader's action, never on a page default.
  function set(kind, key) {
    if (!key) return;
    const s = stored();
    s[kind] = key;
    try {
      root.localStorage.setItem(STORE, JSON.stringify(s));
    } catch (e) {
      /* private mode: the URL still carries the pick */
    }
    try {
      const url = new URL(root.location.href);
      url.searchParams.set("pick", kind + ":" + key);
      root.history.replaceState(root.history.state, "", url);
    } catch (e) {
      /* file:// or sandboxed frame */
    }
    render(kind, key);
  }

  let rows = null;
  let loading = null;
  function load() {
    if (rows) return Promise.resolve(rows);
    if (!loading) {
      loading = fetch(DATA_URL)
        .then((r) => (r.ok ? r.json() : { data: [] }))
        .then((d) => (rows = d.data || d.rows || []))
        .catch(() => (rows = []));
    }
    return loading;
  }

  function find(kind, key) {
    return (rows || []).find((r) => r.kind === kind && r.model_key === key) || null;
  }

  function verdictText(r) {
    if (!r.on_residuals) return "Not ranked · " + (r.reason || r.coverage_status);
    return (r.consensus_verdict || "Near prediction") + " (" + pct(r.residual_pct) + " vs OLS)";
  }

  function panelHtml(kind, r) {
    const here = (root.location.pathname.split("/").pop() || "index.html").replace(/^$/, "index.html");
    const value = r.on_value_ranking && r.value_rank
      ? "#" + r.value_rank + " of " + r.value_n + " · " + r.score_per_dollar.toFixed(2) + " Blender pts/$ at " + usd(r.price_usd) + (r.price_date ? " (" + esc(r.price_date) + ")" : "")
      : "Not on the ranking (no Blender score with a current price)";
    const fair = r.fair_price_usd != null
      ? usd(r.fair_price_usd) + " estimated · listing " + pct(r.fair_gap_pct) + " vs estimate"
      : "No fair-price estimate";
    const cells = [
      ["Value ranking", value, PAGES[0][0]],
      ["Fair price", fair, PAGES[1][0]],
      ["Over & under performers", verdictText(r), PAGES[2][0]],
    ]
      .map(([label, text, page]) => {
        const name = page === here ? "<b>" + label + "</b> (this page)" : '<a href="' + esc(href(page, kind, r.model_key)) + '">' + label + "</a>";
        return "<div><span>" + name + "</span><span>" + esc(text) + "</span></div>";
      })
      .join("");
    return (
      '<div class="tvp-head"><span class="tvp-kicker">Selected across Tech Value</span><b>' + esc(r.name) + "</b>" +
      '<span class="tvp-badge">' + esc(r.coverage_status) + "</span></div>" +
      '<div class="tvp-grid">' + cells + "</div>"
    );
  }

  function render(kind, key) {
    const targets = [...document.querySelectorAll(".tv-pick-panel")].filter((el) => !el.dataset.kind || el.dataset.kind === kind);
    if (!targets.length || !key) return;
    load().then(() => {
      const r = find(kind, key);
      targets.forEach((el) => {
        el.innerHTML = r ? panelHtml(kind, r) : "";
        el.hidden = !r;
      });
    });
  }

  const CSS =
    ".tv-pick-panel{margin:12px 0;padding:10px 14px;border:1px solid rgba(128,140,155,.35);border-radius:12px;font-size:.85rem;line-height:1.45}" +
    ".tvp-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;margin-bottom:6px}" +
    ".tvp-kicker{font-size:.75rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;opacity:.7}" +
    ".tvp-badge{padding:.05em .6em;border-radius:999px;border:1px solid rgba(128,140,155,.45);font-size:.78rem}" +
    ".tvp-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px 18px}" +
    ".tvp-grid div{display:flex;flex-direction:column;min-width:0}" +
    ".tvp-grid div span:last-child{opacity:.8;overflow-wrap:anywhere}" +
    "@media (max-width:700px){.tvp-grid{grid-template-columns:minmax(0,1fr)}}";
  if (typeof document !== "undefined" && !document.getElementById("tvp-style")) {
    const st = document.createElement("style");
    st.id = "tvp-style";
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  root.TVPick = { get, set, urlKind, render, load, find, href };
})(typeof window !== "undefined" ? window : this);
