/* Research dashboard — vanilla JS.
 * Reads window.DASHBOARD_DATA (written by scripts/build_dashboard.py) and
 * renders the static shell in dashboard/index.html.  No network access.
 */
(() => {
  "use strict";

  const DATA = window.DASHBOARD_DATA;
  const MODELS = ["global", "attention", "object"];
  const MODEL_SHORT = { global: "Global", attention: "Attention", object: "Object-aware" };
  const MODEL_FULL = {
    global: "Global CNN-LSTM",
    attention: "Attention (Bahdanau)",
    object: "Object-aware",
  };
  const MODEL_COLORS = { global: "#4C72B0", attention: "#DD8452", object: "#55A868" };
  const FALLBACK = "#8e897d";
  const SECTIONS = [
    ["overview", "Overview"],
    ["pipeline", "Pipeline"],
    ["dataset", "Dataset"],
    ["models", "Models"],
    ["hallucination", "Hallucination"],
    ["bleu-chair", "BLEU vs CHAIR"],
    ["attention", "Attention"],
    ["gallery", "Gallery"],
    ["report", "Report"],
  ];

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, opts = {}, children = []) {
    const node = document.createElement(tag);
    if (opts.class) node.className = opts.class;
    if (opts.text != null) node.textContent = String(opts.text);
    if (opts.attrs) {
      Object.entries(opts.attrs).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== false) {
          node.setAttribute(key, String(value));
        }
      });
    }
    if (opts.style) {
      Object.entries(opts.style).forEach(([key, value]) => {
        if (key.startsWith("--")) node.style.setProperty(key, String(value));
        else node.style[key] = value;
      });
    }
    children.filter(Boolean).forEach((child) => node.appendChild(child));
    return node;
  }

  const SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs = {}) {
    const node = document.createElementNS(SVGNS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function clear(node) {
    if (node) while (node.firstChild) node.removeChild(node.firstChild);
  }

  function num(value, digits = 2, suffix = "") {
    const n = Number(value);
    if (value === null || value === undefined || !Number.isFinite(n)) return "—";
    return `${n.toFixed(digits)}${suffix}`;
  }

  function pct(value, digits = 2) {
    return num(value, digits, "%");
  }

  function compact(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
    if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return `${Math.round(n)}`;
  }

  function captionTotal(dataset) {
    const counts = dataset?.captions || {};
    const known = Object.values(counts).map(Number).filter(Number.isFinite);
    if (!known.length) return "—";
    return compact(known.reduce((sum, n) => sum + n, 0));
  }

  function trainTime(minutes) {
    const n = Number(minutes);
    if (minutes === null || minutes === undefined || !Number.isFinite(n)) return "—";
    return `${n.toFixed(1)} min`;
  }

  function fileSize(bytes) {
    const n = Number(bytes);
    if (!Number.isFinite(n) || n <= 0) return "—";
    if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(n >= 1024 * 100 ? 0 : 1)} KB`;
    return `${n} B`;
  }

  function pValue(value) {
    const n = Number(value);
    if (value === null || value === undefined || !Number.isFinite(n)) return "—";
    if (n === 0) return "< 0.001";
    if (n < 0.001) {
      // Scientific notation for tiny p-values, e.g. "8.1e-39".
      return n.toExponential(1).replace("e", "e");
    }
    return n.toFixed(3);
  }

  function color(id) {
    const model = (DATA.models || []).find((item) => item.id === id);
    return (model && model.color) || MODEL_COLORS[id] || FALLBACK;
  }

  function label(id) {
    const model = (DATA.models || []).find((item) => item.id === id);
    return (model && model.short) || MODEL_SHORT[id] || id;
  }

  function barFill(node, value, max, color) {
    const share = max > 0 ? Math.max(0, Math.min(100, (Number(value) / max) * 100)) : 0;
    // Use the actual width as well as the custom property: some rules read
    // var(--w), while inline width guarantees the bar is visible everywhere.
    node.style.width = `${share}%`;
    node.style.setProperty("--w", `${share}%`);
    if (color) node.style.setProperty("--c", color);
    return node;
  }

  // ── Boot ───────────────────────────────────────────────────────────────
  function boot() {
    if (!DATA || typeof DATA !== "object") {
      const error = $("#error-panel");
      const shell = $("#shell");
      if (error) error.hidden = false;
      if (shell) shell.hidden = true;
      const detail = $("#error-detail");
      if (detail) {
        detail.textContent =
          "dashboard/data/dashboard-data.js is missing or empty. Run `python scripts/build_dashboard.py`, then reload.";
      }
      return;
    }
    const error = $("#error-panel");
    const shell = $("#shell");
    if (error) error.hidden = true;
    if (shell) shell.hidden = false;
    render();
  }

  function render() {
    renderMasthead();
    renderNav();
    renderOverview();
    renderPipeline();
    renderDataset();
    renderModels();
    renderHallucination();
    renderBleuChair();
    renderAttention();
    renderGallery();
    renderReport();
    wirePrint();
    wireLightbox();
    wireImageFallbacks();
    wireScrollSpy();
    document.title = `${DATA.meta.title} · Research dashboard`;
  }

  // ── Masthead / nav ─────────────────────────────────────────────────────
  function renderMasthead() {
    const meta = $("#generated-at");
    if (meta) {
      const generated = DATA.meta.generated_at ? new Date(DATA.meta.generated_at) : new Date();
      meta.textContent = `Generated ${generated.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "2-digit",
      })} · seed ${DATA.meta.seed ?? "—"}`;
    }
  }

  let navLinks = [];
  function renderNav() {
    const nav = $("#nav-list");
    if (!nav) return;
    clear(nav);
    navLinks = [];
    SECTIONS.forEach(([id, text], index) => {
      const anchor = el("a", {
        class: "site-nav__link",
        attrs: { href: `#${id}`, "data-section": id },
      }, [
        el("span", { class: "site-nav__num", text: String(index + 1).padStart(2, "0") }),
        el("span", { text }),
      ]);
      anchor.addEventListener("click", (event) => {
        const target = document.getElementById(id);
        if (!target) return;
        event.preventDefault();
        // scrollIntoView would tuck the heading under the sticky masthead;
        // compute the offset from the CSS scroll-padding instead.
        const offset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
        const top = window.scrollY + target.getBoundingClientRect().top - offset;
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        window.scrollTo({ top: Math.max(0, top), behavior: reduce ? "auto" : "smooth" });
        history.replaceState(null, "", `#${id}`);
      });
      if (index === 0) anchor.classList.add("is-active");
      nav.appendChild(el("li", {}, [anchor]));
      navLinks.push(anchor);
    });
    const topNav = $("#site-nav");
    if (topNav) topNav.hidden = false;
  }

  function wireScrollSpy() {
    const sections = SECTIONS.map(([id]) => document.getElementById(id)).filter(Boolean);
    if (!sections.length) return;
    const setActive = (id) => {
      navLinks.forEach((anchor) => {
        const active = anchor.dataset.section === id;
        anchor.classList.toggle("is-active", active);
        if (active) anchor.setAttribute("aria-current", "location");
        else anchor.removeAttribute("aria-current");
      });
    };
    let ticking = false;
    const update = () => {
      ticking = false;
      const offset = (document.querySelector(".masthead")?.offsetHeight || 0) + 96;
      let current = sections[0].id;
      sections.forEach((section) => {
        if (section.getBoundingClientRect().top - offset <= 0) current = section.id;
      });
      // At the very bottom, force the last section active.
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
        current = sections[sections.length - 1].id;
      }
      setActive(current);
    };
    window.addEventListener(
      "scroll",
      () => {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(update);
      },
      { passive: true },
    );
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  // ── Overview ───────────────────────────────────────────────────────────
  function renderOverview() {
    const { meta, headline } = DATA;
    const kicker = $("#overview-kicker");
    if (kicker) kicker.textContent = `${meta.title} · object hallucination · COCO 2017`;
    const subtitle = $("#overview-subtitle");
    if (subtitle) subtitle.textContent = meta.subtitle || "";
    const question = $("#research-question");
    if (question) question.textContent = headline.question || "";
    const answer = $("#research-answer");
    if (answer) answer.textContent = headline.answer || "";

    const facts = $("#overview-facts");
    if (facts) {
      clear(facts);
      [
        ["Images", compact(meta.n_total)],
        ["Test", compact(meta.n_test)],
        ["Classes", compact((DATA.models || []).length ? (DATA.meta?.n_classes ?? 80) : DATA.meta?.n_classes)],
        ["Seed", meta.seed ?? "—"],
      ].forEach(([key, value]) => {
        facts.appendChild(
          el("div", { class: "fact" }, [
            el("span", { class: "fact__k", text: key }),
            el("span", { class: "fact__v", text: String(value) }),
          ]),
        );
      });
    }

    const best = (DATA.models || []).find((model) => model.id === headline.best_model);
    const kpis = [
      ["Best BLEU-4", num(headline.best_bleu4, 2), best ? best.label : "test split", "#4C72B0"],
      ["Lowest CHAIRs", pct(headline.best_chair_s, 2), "object-aware model", "#55A868"],
      ["CHAIRs reduction", `${num(headline.hallucination_reduction_pp, 2)} pp`, "global → object-aware", "#55A868"],
      ["Pooled ρ", num(headline.pooled_rho, 3), `Spearman · n = ${compact(DATA.correlations?.pooled?.n)} captions`, "#DD8452"],
    ];
    const grid = $("#kpi-grid");
    if (grid) {
      clear(grid);
      kpis.forEach(([k, v, note, accent]) => {
        const card = el("article", { class: "kpi", style: { "--kpi-color": accent } }, [
          el("p", { class: "kpi__label", text: k }),
          el("p", { class: "kpi__value", text: String(v) }),
          el("p", { class: "kpi__meta", text: note }),
        ]);
        grid.appendChild(card);
      });
    }

    const glance = $("#glance-bars");
    if (glance) {
      clear(glance);
      const groups = [
        ["BLEU-4 (higher is better)", "bleu4", 25],
        ["CHAIRs (lower is better)", "chair_s", 70],
      ];
      groups.forEach(([title, key, max]) => {
        const group = el("div", { class: "glance__group" }, [
          el("div", { class: "glance__title" }, [el("span", { text: title })]),
        ]);
        (DATA.models || []).forEach((model) => {
          const fill = el("div", {
            class: "glance__fill",
            style: { "--c": model.color || color(model.id) },
          });
          barFill(fill, model[key], max);
          group.appendChild(
            el("div", { class: "glance__row" }, [
              el("span", { class: "glance__name", text: model.short || label(model.id) }),
              el("div", { class: "glance__track" }, [fill]),
              el("span", { class: "glance__val", text: num(model[key], 2) }),
            ]),
          );
        });
        glance.appendChild(group);
      });
    }

    const notes = $("#reading-notes");
    if (notes) {
      clear(notes);
      [
        "CHAIRs is the share of unique object mentions that are absent from COCO annotations; lower is better.",
        "BLEU-4 is cumulative 4-gram overlap with five references; higher is better.",
        "Model-level agreement can hide caption-level disagreement — see matched BLEU.",
        "COCO annotations are incomplete, so CHAIR is an upper bound on true hallucination.",
      ].forEach((text) => notes.appendChild(el("li", { text })));
    }
  }

  // ── Pipeline ───────────────────────────────────────────────────────────
  function renderPipeline() {
    const tasks = $("#task-coverage");
    if (tasks) {
      clear(tasks);
      (DATA.tasks || []).forEach((task, index) => {
        const evidence = el("span", { class: "task__evidence" });
        (task.evidence || []).forEach((item) => {
          evidence.appendChild(
            el("a", { text: item.label || item.path, attrs: { href: item.path } }),
          );
          evidence.appendChild(document.createTextNode(" "));
        });
        tasks.appendChild(
          el("article", { class: "task" }, [
            el("div", { class: "task__top" }, [
              el("span", { class: "task__id", text: String(index + 1).padStart(2, "0") }),
              el("span", {
                class: "chip-status",
                text: task.status || "complete",
                attrs: { "data-status": task.status || "complete" },
              }),
            ]),
            el("h3", { class: "task__label", text: task.label }),
            el("p", { class: "task__desc", text: task.description || "" }),
            el("p", { class: "task__metric", text: task.metric || "" }),
            evidence,
          ]),
        );
      });
    }

    const stepper = $("#pipeline-stepper");
    if (stepper) {
      clear(stepper);
      (DATA.pipeline || []).forEach((step) => {
        stepper.appendChild(
          el("li", { class: "step" }, [
            el("a", { class: "step__label", text: step.label, attrs: { href: step.file || "#" } }),
            el("p", { class: "step__detail", text: step.detail || "" }),
            el("div", { class: "step__foot" }, [
              el("span", { class: "step__runtime", text: step.runtime || "—" }),
              el("a", { class: "step__file", text: (step.file || "").replace("../", ""), attrs: { href: step.file || "#" } }),
            ]),
          ]),
        );
      });
    }
  }

  // ── Dataset ────────────────────────────────────────────────────────────
  function renderDataset() {
    const ds = DATA.dataset || {};
    const dek = $("#dataset-dek");
    if (dek) {
      const nTotal = Number(DATA.meta?.n_total);
      dek.textContent = Number.isFinite(nTotal) && nTotal > 0
        ? `${compact(nTotal)} COCO 2017 validation images, re-split deterministically. Captions and instance annotations drive both quality and grounding metrics.`
        : "COCO 2017 validation images, re-split deterministically. Captions and instance annotations drive both quality and grounding metrics.";
    }

    const strip = $("#dataset-strip");
    if (strip) {
      clear(strip);
      [
        ["Images", compact(DATA.meta.n_total), "COCO 2017 val"],
        ["Train / val / test", `${compact(ds.splits?.train)} / ${compact(ds.splits?.val)} / ${compact(ds.splits?.test)}`, "images"],
        ["Captions", captionTotal(ds), "five per image"],
        ["Vocabulary", compact(DATA.meta.vocab_size), "min frequency 3"],
      ].forEach(([key, value, sub]) => {
        strip.appendChild(
          el("div", { class: "stat" }, [
            el("span", { class: "stat__k", text: key }),
            el("span", { class: "stat__v", text: String(value) }),
            el("span", { class: "stat__sub", text: sub }),
          ]),
        );
      });
    }

    const splitBars = $("#split-bars");
    if (splitBars) {
      clear(splitBars);
      if (!ds.splits || !Object.keys(ds.splits).length) {
        splitBars.appendChild(el("p", { class: "empty-note", text: "No split counts in this payload." }));
      }
      const total = Object.values(ds.splits || {}).reduce((sum, n) => sum + Number(n || 0), 0) || 1;
      const colors = { train: "#4C72B0", val: "#DD8452", test: "#55A868" };
      Object.entries(ds.splits || {}).forEach(([name, count]) => {
        const share = (Number(count) / total) * 100;
        const fill = el("div", {
          class: "split-bar__fill",
          style: { "--c": colors[name] || "#777" },
        });
        barFill(fill, count, total);
        splitBars.appendChild(
          el("div", { class: "split-bar" }, [
            el("div", { class: "split-bar__top" }, [
              el("span", { class: "split-bar__name", text: name }),
              el("span", { class: "split-bar__val", text: compact(count) }),
              el("span", { class: "split-bar__pct", text: `${share.toFixed(1)}%` }),
            ]),
            el("div", { class: "split-bar__track" }, [fill]),
          ]),
        );
      });
    }

    const splitTable = $("#split-table");
    if (splitTable) {
      clear(splitTable);
      if (!ds.splits || !Object.keys(ds.splits).length) {
        // Nothing to tabulate; the bars already explain the gap.
        splitTable.appendChild(el("p", { class: "empty-note", text: "No split table available." }));
      } else {
        const table = el("table", { class: "data-table data-table--compact" }, [
          el("thead", {}, [
            el("tr", {}, [
              el("th", { text: "Split" }),
              el("th", { text: "Images" }),
              el("th", { text: "Captions" }),
            ]),
          ]),
        ]);
        const tbody = el("tbody");
        Object.keys(ds.splits || {}).forEach((name) => {
          tbody.appendChild(
            el("tr", {}, [
              el("th", { text: name, attrs: { scope: "row" } }),
              el("td", { class: "num", text: compact(ds.splits[name]) }),
              el("td", { class: "num", text: compact(ds.captions?.[name]) }),
            ]),
          );
        });
        table.appendChild(tbody);
        splitTable.appendChild(el("div", { class: "table-wrap" }, [table]));
      }
    }

    const captionStats = $("#caption-stats");
    if (captionStats) {
      clear(captionStats);
      const cl = ds.caption_length || {};
      [
        ["Mean tokens", num(cl.mean, 1), "per caption"],
        ["Median", num(cl.p50, 0), "tokens"],
        ["P95", num(cl.p95, 0), "tokens"],
        ["Maximum", num(cl.max, 0), "tokens"],
        ["Objects / image", num(ds.objects_per_image?.mean, 2), "COCO classes"],
        ["Empty images", compact(ds.detector?.empty_images), "no detections"],
      ].forEach(([key, value, sub]) => {
        captionStats.appendChild(
          el("div", { class: "mini-stat" }, [
            el("span", { class: "mini-stat__k", text: key }),
            el("span", { class: "mini-stat__v", text: String(value) }),
            el("span", { class: "mini-stat__sub", text: sub }),
          ]),
        );
      });
    }

    const topObjects = $("#top-objects");
    if (topObjects) {
      clear(topObjects);
      const items = (ds.top_objects || []).slice(0, 10);
      if (!items.length) topObjects.appendChild(el("p", { class: "empty-note", text: "No object counts in this payload." }));
      const max = Math.max(1, ...items.map((item) => Number(item.count) || 0));
      items.forEach((item) => {
        const fill = el("div", { class: "bar-row__fill" });
        barFill(fill, item.count, max);
        topObjects.appendChild(
          el("div", { class: "bar-row" }, [
            el("span", { class: "bar-row__name", text: item.name }),
            el("div", { class: "bar-row__track" }, [fill]),
            el("span", { class: "bar-row__val", text: compact(item.count) }),
          ]),
        );
      });
    }

    const det = $("#detector-stats");
    if (det) {
      clear(det);
      const d = ds.detector || {};
      [
        ["Model", d.name || "—"],
        ["Score threshold", num(d.threshold, 2)],
        ["Max detections", num(d.max_detections, 0)],
        ["Mean detections", num(d.mean_detections, 2)],
        ["Empty images", compact(d.empty_images)],
        ["Runtime", d.seconds ? `${(Number(d.seconds) / 60).toFixed(1)} min` : "—"],
      ].forEach(([key, value]) => {
        det.appendChild(el("dt", { text: key }));
        det.appendChild(el("dd", { text: String(value) }));
      });
    }

    const notes = $("#dataset-notes");
    if (notes) {
      clear(notes);
      [
        "This is a deterministic re-split of COCO val2017, not the official train/val partition.",
        Number.isFinite(Number(ds.detector?.empty_images))
          ? `${compact(ds.detector?.empty_images)} images have no annotations or detections, so their ground-truth object set is empty.`
          : "Some images have no annotations or detections, so their ground-truth object set is empty.",
        "The detector is COCO-pretrained and independent of the captioners; its labels are inputs, not targets.",
        "A visible object missing from annotations counts as hallucinated, so CHAIR overestimates.",
      ].forEach((text) => notes.appendChild(el("li", { text })));
    }
  }

  // ── Models ─────────────────────────────────────────────────────────────
  let sortKey = "bleu4";
  let sortDir = -1;

  function renderModels() {
    const dek = $("#models-dek");
    if (dek) {
      dek.textContent =
        "Same ResNet-18 features, LSTM decoder, optimiser and training budget. Only the information available to the decoder changes.";
    }
    const grid = $("#model-cards");
    if (grid) {
      clear(grid);
      (DATA.models || []).forEach((model) => {
        const stats = el("div", { class: "model-card__stats" });
        [
          ["BLEU-4", num(model.bleu4, 2)],
          ["CHAIRs", pct(model.chair_s, 2)],
          ["CIDEr-D", num(model.cider_d, 3)],
          ["Object F1", pct(model.object_f1, 1)],
          ["Parameters", model.params_label || compact(model.params)],
          ["Train time", trainTime(model.train_minutes)],
        ].forEach(([key, value]) => {
          stats.appendChild(
            el("div", { class: "model-stat" }, [
              el("span", { class: "model-stat__k", text: key }),
              el("span", { class: "model-stat__v", text: String(value) }),
            ]),
          );
        });
        grid.appendChild(
          el("article", { class: "model-card", style: { "--c": model.color || color(model.id) } }, [
            el("div", { class: "model-card__head" }, [
              el("span", { class: "model-card__short", text: model.short || label(model.id) }),
              el("h3", { class: "model-card__label", text: model.label }),
            ]),
            el("p", { class: "model-card__desc", text: model.description || "" }),
            el("div", { class: "model-card__tags" }, [
              el("span", { class: `tag ${model.uses_attention ? "tag--on" : ""}`, text: model.uses_attention ? "attention" : "no attention" }),
              el("span", { class: `tag ${model.uses_objects ? "tag--on" : ""}`, text: model.uses_objects ? "object labels" : "no object labels" }),
            ]),
            stats,
            el("div", { class: "model-card__foot" }, [
              el("span", { text: trainTime(model.train_minutes) }),
              el("span", { text: model.params_label || compact(model.params) }),
            ]),
          ]),
        );
      });
    }

    const table = $("#metrics-table");
    if (table) {
      const head = $("thead", table);
      clear(head);
      // Keep the metric order used by the payload's models entries so that
      // table columns and data keys stay in one place.
      const columns = [
        ["model", "Model"],
        ["bleu1", "BLEU-1"],
        ["bleu2", "BLEU-2"],
        ["bleu3", "BLEU-3"],
        ["bleu4", "BLEU-4"],
        ["cider_d", "CIDEr-D"],
        ["chair_s", "CHAIRs"],
        ["chair_i", "CHAIRi"],
        ["object_precision", "Obj. P"],
        ["object_recall", "Obj. R"],
        ["object_f1", "Obj. F1"],
        ["params_label", "Params"],
        ["train_minutes", "Train"],
      ];
      const row = el("tr");
      columns.forEach(([key, text]) => {
        const button = el("button", { class: "th-sort", attrs: { type: "button" } }, [
          el("span", { text }),
          el("span", { class: "th-sort__icon", text: "↕", attrs: { "aria-hidden": "true", "data-sort-icon": key } }),
        ]);
        const th = el("th", { attrs: { scope: "col", "data-sort-key": key } }, [button]);
        button.addEventListener("click", () => setSort(key));
        row.appendChild(th);
      });
      head.appendChild(row);
      syncAriaSort();

      const select = $("#metrics-sort");
      if (select) {
        select.value = sortKey;
        select.addEventListener("change", () => setSort(select.value));
      }
      const dir = $("#metrics-dir");
      if (dir) {
        dir.addEventListener("click", () => {
          sortDir *= -1;
          updateSortUi();
          renderMetricRows();
        });
      }
      renderMetricRows();
    }

    renderMetricBars();
    renderTrainingCurves();
  }

  function setSort(key) {
    if (key === sortKey) sortDir *= -1;
    else {
      sortKey = key;
      sortDir = key === "model" || key === "params_label" ? 1 : -1;
    }
    const select = $("#metrics-sort");
    if (select && select.value !== key) select.value = key;
    updateSortUi();
    renderMetricRows();
  }

  function updateSortUi() {
    const icon = $("#metrics-dir-icon");
    const text = $("#metrics-dir-label");
    const dir = $("#metrics-dir");
    if (icon) icon.textContent = sortDir === 1 ? "↑" : "↓";
    if (text) text.textContent = sortDir === 1 ? "Ascending" : "Descending";
    if (dir) dir.setAttribute("aria-pressed", String(sortDir === 1));
    $$("#metrics-table [data-sort-icon]").forEach((icon) => {
      icon.textContent = icon.dataset.sortIcon === sortKey ? (sortDir === 1 ? "↑" : "↓") : "↕";
    });
    syncAriaSort();
  }

  // aria-sort belongs on the sorted column only; every other header must say
  // "none" explicitly rather than relying on the implicit default.
  function syncAriaSort() {
    $$("#metrics-table thead th").forEach((th) => {
      const active = th.dataset.sortKey === sortKey;
      if (active) th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
      else th.setAttribute("aria-sort", "none");
    });
  }

  // Lower is better for CHAIR and for nothing else in this table.
  const LOWER_IS_BETTER = new Set(["chair_s", "chair_i"]);

  function metricText(model, key) {
    switch (key) {
      case "cider_d": return num(model.cider_d, 3);
      case "params_label": return model.params_label || compact(model.params);
      case "train_minutes": return trainTime(model.train_minutes);
      case "model": return model.short || label(model.id);
      default: return num(model[key], 2);
    }
  }

  function renderMetricRows() {
    const body = $("#metrics-table tbody");
    if (!body) return;
    clear(body);
    const rows = (DATA.models || []).slice();
    // Array.prototype.sort is stable in every browser we support, so ties keep
    // the payload order (global, attention, object) rather than shuffling.
    rows.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string" || typeof bv === "string") {
        return String(av).localeCompare(String(bv)) * sortDir;
      }
      return ((Number(av) || 0) - (Number(bv) || 0)) * sortDir;
    });

    const bestByKey = {};
    ["bleu4", "cider_d", "chair_s", "chair_i", "object_f1"].forEach((key) => {
      const values = rows.map((m) => Number(m[key])).filter(Number.isFinite);
      if (!values.length) return;
      bestByKey[key] = LOWER_IS_BETTER.has(key) ? Math.min(...values) : Math.max(...values);
    });

    rows.forEach((model) => {
      const cells = [
        el("th", { attrs: { scope: "row" } }, [
          el("span", { class: "model-dot", style: { "--c": model.color || color(model.id) } }),
          el("span", { text: model.short || label(model.id) }),
        ]),
      ];
      [
        "bleu1", "bleu2", "bleu3", "bleu4", "cider_d",
        "chair_s", "chair_i", "object_precision", "object_recall", "object_f1",
        "params_label", "train_minutes",
      ].forEach((key) => {
        const raw = model[key];
        const missing = raw === null || raw === undefined || (key !== "params_label" && !Number.isFinite(Number(raw)));
        const isBest = !missing && key in bestByKey && Number(raw) === bestByKey[key];
        cells.push(el("td", {
          class: [isBest ? "cell-best" : "", missing ? "cell-unavailable" : ""].filter(Boolean).join(" "),
          text: metricText(model, key),
        }));
      });
      body.appendChild(el("tr", {}, cells));
    });

    const status = $("#metrics-status");
    if (status) {
      const column = ($$("#metrics-table thead th").find((th) => th.dataset.sortKey === sortKey)?.textContent || sortKey).trim();
      const arrow = sortDir === 1 ? "ascending" : "descending";
      status.textContent = `Sorted by ${column.replace(/[↕↑↓]/g, "").trim()} (${arrow}). Bold marks the best value in each column — for CHAIR, lower is best.`;
    }
  }

  function renderMetricBars() {
    const root = $("#model-metric-bars");
    if (!root) return;
    clear(root);
    [
      ["bleu4", "BLEU-4", "higher is better"],
      ["chair_s", "CHAIRs", "lower is better"],
      ["cider_d", "CIDEr-D", "higher is better"],
      ["object_f1", "Object F1", "higher is better"],
    ].forEach(([key, title, note]) => {
      const values = (DATA.models || []).map((m) => Number(m[key])).filter(Number.isFinite);
      if (!values.length) return;
      const max = Math.max(0.0001, ...values);
      const group = el("div", { class: "metric-group" }, [
        el("div", { class: "metric-group__title" }, [
          el("span", { text: title }),
          el("span", { text: note }),
        ]),
      ]);
      (DATA.models || []).forEach((model) => {
        const fill = el("div", { class: "glance__fill", style: { "--c": model.color || color(model.id) } });
        barFill(fill, model[key], max);
        group.appendChild(
          el("div", { class: "glance__row" }, [
            el("span", { class: "glance__name", text: model.short || label(model.id) }),
            el("div", { class: "glance__track" }, [fill]),
            el("span", { class: "glance__val", text: num(model[key], 2) }),
          ]),
        );
      });
      root.appendChild(group);
    });
  }

  function renderTrainingCurves() {
    const root = $("#training-curves");
    if (!root) return;
    clear(root);
    const training = DATA.training || {};
    const ids = MODELS.filter((id) => Array.isArray(training[id]) && training[id].length);
    if (!ids.length) {
      root.appendChild(el("p", { class: "empty-note", text: "No training history in this payload." }));
      return;
    }
    const width = 560;
    const height = 250;
    const pad = { l: 44, r: 14, t: 14, b: 30 };
    const chart = svg("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "curve-svg",
      role: "img",
      "aria-label": "Validation loss by epoch for three models",
    });
    const epochs = Math.max(...ids.map((id) => training[id].length));
    const x = (epoch) => pad.l + ((epoch - 1) / Math.max(1, epochs - 1)) * (width - pad.l - pad.r);
    const losses = ids.flatMap((id) => training[id].map((row) => Number(row.val_loss)));
    const maxLoss = Math.max(1, ...losses.filter(Number.isFinite)) * 1.06;
    const y = (value) => pad.t + (1 - value / maxLoss) * (height - pad.t - pad.b);
    for (let i = 0; i <= 4; i += 1) {
      const value = (maxLoss / 4) * i;
      const yy = y(value);
      chart.appendChild(svg("line", { class: "curve-grid", x1: pad.l, x2: width - pad.r, y1: yy, y2: yy }));
      const text = svg("text", { class: "curve-axis", x: pad.l - 8, y: yy + 3, "text-anchor": "end" });
      text.textContent = value.toFixed(1);
      chart.appendChild(text);
    }
    ids.forEach((id) => {
      const points = training[id]
        .filter((row) => Number.isFinite(Number(row.val_loss)))
        .map((row) => `${x(row.epoch)},${y(Number(row.val_loss))}`)
        .join(" ");
      chart.appendChild(
        svg("polyline", { class: "curve-line", points, stroke: color(id) }),
      );
    });
    for (let epoch = 1; epoch <= epochs; epoch += 2) {
      const text = svg("text", { class: "curve-axis", x: x(epoch), y: height - 8, "text-anchor": "middle" });
      text.textContent = String(epoch);
      chart.appendChild(text);
    }
    root.appendChild(chart);
    const legend = $("#training-legend");
    if (legend) {
      clear(legend);
      ids.forEach((id) => {
        legend.appendChild(
          el("span", { class: "legend__item" }, [
            el("span", { class: "legend__swatch", style: { background: color(id) } }),
            el("span", { text: label(id) }),
          ]),
        );
      });
    }
  }

  // ── Hallucination ──────────────────────────────────────────────────────
  function renderHallucination() {
    const root = $("#chair-compare");
    if (root) {
      clear(root);
      (DATA.models || []).forEach((model) => {
        root.appendChild(
          el("div", { class: "chair-block" }, [
            el("div", { class: "chair-block__title" }, [
              el("span", { text: model.short || label(model.id) }),
              el("span", { text: `CHAIRs ${num(model.chair_s, 2)}% · CHAIRi ${num(model.chair_i, 2)}%` }),
            ]),
            el("div", { class: "glance__row" }, [
              el("span", { class: "glance__name", text: "CHAIRs" }),
              el("div", { class: "glance__track" }, [
                barFill(
                  el("div", { class: "glance__fill", style: { background: model.color || color(model.id) } }),
                  model.chair_s,
                  70,
                ),
              ]),
              el("span", { class: "glance__val", text: num(model.chair_s, 2) }),
            ]),
            el("div", { class: "glance__row" }, [
              el("span", { class: "glance__name", text: "CHAIRi" }),
              el("div", { class: "glance__track" }, [
                barFill(
                  el("div", { class: "glance__fill", style: { background: "#b9b3a7" } }),
                  model.chair_i,
                  70,
                ),
              ]),
              el("span", { class: "glance__val", text: num(model.chair_i, 2) }),
            ]),
          ]),
        );
      });
    }

    const top = $("#top-hallucinated");
    if (top) {
      clear(top);
      const items = (DATA.hallucination?.top_hallucinated || []).slice(0, 10);
      if (!items.length) top.appendChild(el("p", { class: "empty-note", text: "No hallucination counts in this payload." }));
      const max = Math.max(1, ...items.map((item) => Number(item.count) || 0));
      items.forEach((item) => {
        top.appendChild(
          el("div", { class: "bar-row" }, [
            el("span", { class: "bar-row__name", text: item.name }),
            el("div", { class: "bar-row__track" }, [barFill(el("div", { class: "bar-row__fill" }), item.count, max)]),
            el("span", { class: "bar-row__val", text: compact(item.count) }),
          ]),
        );
      });
    }

    const examples = $("#hallucination-examples");
    if (examples) {
      clear(examples);
      const list = DATA.hallucination?.examples || [];
      if (!list.length) {
        examples.appendChild(el("p", { class: "empty-note", text: "No worked examples in this payload." }));
      }
      list.forEach((example) => examples.appendChild(exampleCard(example)));
    }

    const legend = $("#hallucination-legend");
    if (legend) {
      clear(legend);
      legend.appendChild(el("span", { class: "legend__item" }, [
        el("span", { class: "halluc-tag", text: "hallucinated" }),
        el("span", { text: "not in COCO annotations" }),
      ]));
    }
  }

  function highlightCaption(text, words) {
    const bad = [...new Set((words || []).map((word) => String(word).toLowerCase()))].sort(
      (a, b) => b.length - a.length,
    );
    if (!bad.length) return [el("span", { text })];
    const escaped = bad.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");
    const parts = [];
    let last = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) parts.push(document.createTextNode(text.slice(last, match.index)));
      parts.push(el("mark", { class: "hallucinated", text: match[0] }));
      last = match.index + match[0].length;
    }
    if (last < text.length) parts.push(document.createTextNode(text.slice(last)));
    return parts;
  }

  function exampleCard(example) {
    const figure = el("div", { class: "example-card__figure" }, [
      el("img", {
        attrs: {
          src: example.image,
          alt: `COCO image ${example.image_id}`,
          loading: "lazy",
          decoding: "async",
        },
      }),
      el("span", { class: "example-card__badge", text: `#${example.image_id}` }),
    ]);
    const gt = el("p", { class: "example-card__gt" }, [
      el("span", { class: "example-card__gt-label", text: "GT" }),
      ...(example.gt_objects || []).map((name) => el("span", { class: "gt-tag", text: name })),
    ]);
    if (!(example.gt_objects || []).length) {
      gt.appendChild(el("span", { class: "gt-tag", text: "none" }));
    }
    const captions = el("div", { class: "caption-list" });
    MODELS.forEach((id) => {
      const caption = example.captions?.[id];
      if (!caption) return;
      captions.appendChild(
        el("div", { class: "caption-row" }, [
          el("span", { class: "caption-row__model", text: label(id) }),
          el("span", { class: "caption-row__text" }, highlightCaption(caption.text || "", caption.hallucinated || [])),
        ]),
      );
    });
    const body = el("div", { class: "example-card__body" }, [gt, captions]);
    const card = el("article", { class: "example-card" }, [figure, body]);
    return card;
  }

  // ── BLEU vs CHAIR ──────────────────────────────────────────────────────
  function renderBleuChair() {
    const corr = $("#correlation-cards");
    if (corr) {
      clear(corr);
      if (!Object.keys(DATA.correlations || {}).length) {
        corr.appendChild(el("p", { class: "empty-note", text: "No correlations in this payload." }));
      }
      Object.entries(DATA.correlations || {}).forEach(([id, row]) => {
        const model = (DATA.models || []).find((item) => item.id === id);
        corr.appendChild(
          el("article", {
            class: `corr-card ${id === "pooled" ? "corr-card--pooled" : ""}`,
            style: { "--c": model ? model.color : "#111" },
          }, [
            el("span", { class: "corr-card__label", text: id === "pooled" ? "Pooled" : label(id) }),
            el("span", { class: "corr-card__rho", text: num(row.rho, 3) }),
            el("span", { class: "corr-card__meta", text: `n = ${compact(row.n)} · p = ${pValue(row.p)}` }),
            el("span", { class: "corr-card__rate", text: `Hallucination rate ${pct(row.rate, 1)}` }),
          ]),
        );
      });
    }

    const quartiles = DATA.quartiles || [];
    const table = $("#quartile-table");
    if (table) {
      const head = $("thead", table);
      clear(head);
      head.appendChild(
        el("tr", {}, [
          el("th", { text: "Quartile" }),
          el("th", { text: "N" }),
          el("th", { text: "BLEU-4 range" }),
          el("th", { text: "Halluc. rate" }),
        ]),
      );
      const body = $("tbody", table);
      clear(body);
      if (!quartiles.length) {
        body.appendChild(el("tr", {}, [el("td", { class: "empty-note", text: "No quartiles in this payload.", attrs: { colspan: "4" } })]));
      }
      quartiles.forEach((row) => {
        body.appendChild(
          el("tr", {}, [
            el("th", { text: row.bucket, attrs: { scope: "row" } }),
            el("td", { text: compact(row.n) }),
            el("td", { text: `${num(row.bleu_min, 1)}–${num(row.bleu_max, 1)}` }),
            el("td", { text: pct(row.rate, 1) }),
          ]),
        );
      });
    }

    const qchart = $("#quartile-chart");
    if (qchart) {
      clear(qchart);
      const max = Math.max(1, ...quartiles.map((row) => Number(row.rate) || 0));
      quartiles.forEach((row) => {
        qchart.appendChild(
          el("div", { class: "quartile-col" }, [
            el("div", {
              class: "quartile-col__bar",
              style: { height: `${(Number(row.rate) / max) * 88}px`, "--h": `${(Number(row.rate) / max) * 100}%` },
            }),
            el("span", { class: "quartile-col__label", text: row.bucket }),
          ]),
        );
      });
    }

    renderMatchedChart();
  }

  function renderMatchedChart() {
    const root = $("#matched-chart");
    if (!root) return;
    clear(root);
    const rows = DATA.matched_bleu || [];
    const bins = [...new Set(rows.map((row) => row.bin))];
    if (!bins.length) {
      root.appendChild(el("p", { class: "empty-note", text: "No matched-BLEU bins in this payload." }));
      return;
    }
    const width = 720;
    const height = 290;
    const pad = { l: 44, r: 16, t: 16, b: 40 };
    const chart = svg("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "matched-svg",
      role: "img",
      "aria-label": "Hallucination rate by sentence BLEU bin for three models",
    });
    const max = Math.max(1, ...rows.map((row) => Number(row.rate) || 0)) * 1.1;
    const group = (width - pad.l - pad.r) / bins.length;
    const barW = Math.min(30, group / 4.4);
    const y = (value) => pad.t + (1 - value / max) * (height - pad.t - pad.b);
    for (let i = 0; i <= 4; i += 1) {
      const value = (max / 4) * i;
      const yy = y(value);
      chart.appendChild(svg("line", { class: "mc-grid", x1: pad.l, x2: width - pad.r, y1: yy, y2: yy }));
      const text = svg("text", { class: "mc-axis", x: pad.l - 8, y: yy + 3, "text-anchor": "end" });
      text.textContent = `${Math.round(value)}%`;
      chart.appendChild(text);
    }
    bins.forEach((bin, binIndex) => {
      const models = rows.filter((row) => row.bin === bin);
      const center = pad.l + group * (binIndex + 0.5);
      models.forEach((row, rowIndex) => {
        const value = Number(row.rate) || 0;
        const x = center - (barW * models.length) / 2 + rowIndex * barW;
        const bar = svg("rect", {
          class: "mc-bar",
          x,
          y: y(value),
          width: barW - 3,
          height: Math.max(1, height - pad.b - y(value)),
          fill: color(row.model),
        });
        const title = svg("title");
        title.textContent = `${label(row.model)} · ${bin} · ${num(value, 1)}% (n=${row.n})`;
        bar.appendChild(title);
        chart.appendChild(bar);

        // 95% CI whisker from the payload, when present.
        const ci = Array.isArray(row.ci) ? row.ci.map(Number) : null;
        if (ci && ci.length === 2 && ci.every(Number.isFinite)) {
          const cx = x + (barW - 3) / 2;
          const top = y(Math.min(max, Math.max(ci[0], ci[1])));
          const bottom = y(Math.max(0, Math.min(ci[0], ci[1])));
          const whisker = svg("g", { class: "mc-ci" });
          whisker.appendChild(svg("line", { x1: cx, x2: cx, y1: top, y2: bottom }));
          whisker.appendChild(svg("line", { class: "mc-ci__cap", x1: cx - 3, x2: cx + 3, y1: top, y2: top }));
          whisker.appendChild(svg("line", { class: "mc-ci__cap", x1: cx - 3, x2: cx + 3, y1: bottom, y2: bottom }));
          chart.appendChild(whisker);
        }

        if (binIndex === bins.length - 1 || value >= 20) {
          const valueLabel = svg("text", {
            class: "mc-value",
            x: x + (barW - 3) / 2,
            y: y(value) - 5,
            "text-anchor": "middle",
          });
          valueLabel.textContent = num(value, 0);
          chart.appendChild(valueLabel);
        }
      });
      const text = svg("text", { class: "mc-axis", x: center, y: height - 12, "text-anchor": "middle" });
      text.textContent = bin;
      chart.appendChild(text);
    });
    root.appendChild(chart);

    const legend = $("#matched-legend");
    if (legend) {
      clear(legend);
      MODELS.forEach((id) => {
        legend.appendChild(
          el("span", { class: "legend__item" }, [
            el("span", { class: "legend__swatch", style: { background: color(id) } }),
            el("span", { text: label(id) }),
          ]),
        );
      });
    }

    const insight = $("#matched-insight");
    if (insight) {
      const rates = DATA.headline?.matched_rates || {};
      insight.textContent =
        `In the ${DATA.headline?.matched_bin || "10–20"} BLEU bin, hallucination rates are ` +
        `${num(rates.global, 1)}% (global), ${num(rates.attention, 1)}% (attention) and ` +
        `${num(rates.object, 1)}% (object). Similar BLEU, very different grounding.`;
    }
    wireFigureData();
  }

  // ── Attention ──────────────────────────────────────────────────────────
  const attention = { index: 0, word: null };

  function renderAttention() {
    const examples = DATA.attention_examples || [];
    const list = $("#attention-example-list");
    if (list) {
      clear(list);
      examples.forEach((example, index) => {
        const button = el("button", {
          class: "example-btn",
          attrs: { type: "button", "aria-pressed": String(index === 0) },
        }, [
          el("img", { class: "example-btn__thumb", attrs: { src: example.image, alt: "", loading: "lazy" } }),
          el("span", { class: "example-btn__body" }, [
            el("span", { class: "example-btn__title", text: `Image ${example.image_id}` }),
            el("span", { class: "example-btn__sub", text: example.caption }),
          ]),
        ]);
        button.addEventListener("click", () => selectAttention(index));
        list.appendChild(button);
      });
    }

    const toggle = $("#attention-toggle");
    if (toggle) {
      toggle.addEventListener("click", (event) => {
        const button = event.target.closest("[data-mode]");
        if (!button) return;
        $$("[data-mode]", toggle).forEach((item) => {
          const active = item === button;
          item.classList.toggle("is-active", active);
          item.setAttribute("aria-pressed", String(active));
        });
        updateAttentionImage();
      });
    }

    if (examples.length) selectAttention(0);
    else {
      const caption = $("#attention-caption-text");
      if (caption) caption.textContent = "No attention examples in this payload.";
    }
  }

  function selectAttention(index) {
    const examples = DATA.attention_examples || [];
    const example = examples[index];
    if (!example) return;
    attention.index = index;
    attention.word = (example.highlight && example.highlight[0]) ?? 0;
    $$("#attention-example-list .example-btn").forEach((button, i) => {
      button.setAttribute("aria-pressed", String(i === index));
    });
    const image = $("#attention-image");
    if (image) {
      image.src = example.image;
      image.alt = `COCO image ${example.image_id}`;
    }
    highlightAttentionWord();
    const meta = $("#attention-meta");
    if (meta) {
      meta.textContent = `${example.words.length} generated words · ${(example.overlays || []).length} overlays stored`;
    }

    const chips = $("#attention-chips");
    if (chips) {
      clear(chips);
      example.words.forEach((word, wordIndex) => {
        const hasOverlay = (example.overlays || []).some((overlay) => overlay.index === wordIndex);
        const chip = el("button", {
          class: "word-chip",
          text: word,
          attrs: {
            type: "button",
            "aria-pressed": String(wordIndex === attention.word),
            "data-has-overlay": String(hasOverlay),
            "aria-label": hasOverlay ? word : `${word} (no stored overlay)`,
          },
        });
        // Words without a stored map still become active so the caption
        // highlight and the status line stay in sync; the image falls back
        // to the base photograph.
        chip.addEventListener("click", () => {
          attention.word = wordIndex;
          $$("#attention-chips .word-chip").forEach((item, i) => {
            item.setAttribute("aria-pressed", String(i === wordIndex));
          });
          highlightAttentionWord();
          updateAttentionImage();
        });
        chips.appendChild(chip);
      });
    }

    const thumbs = $("#attention-thumbs");
    if (thumbs) {
      clear(thumbs);
      examples.forEach((item, itemIndex) => {
        const thumb = el("button", {
          class: "thumb-btn",
          attrs: {
            type: "button",
            "aria-pressed": String(itemIndex === index),
            "aria-label": `Example ${itemIndex + 1}: image ${item.image_id}`,
          },
        }, [el("img", { attrs: { src: item.image, alt: "", loading: "lazy" } })]);
        thumb.addEventListener("click", () => selectAttention(itemIndex));
        thumbs.appendChild(thumb);
      });
    }
    updateAttentionImage();
  }

  // Re-render the caption so only the currently selected word is emphasised.
  function highlightAttentionWord() {
    const example = (DATA.attention_examples || [])[attention.index];
    const caption = $("#attention-caption-text");
    if (!example || !caption) return;
    const words = example.words || [];
    const active = words[attention.word];
    if (active == null) {
      caption.textContent = example.caption || "";
      return;
    }
    // Word segmentation must match the payload's `words` array, so split on
    // whitespace and keep punctuation attached instead of tokenising again.
    const tokens = String(example.caption || "").split(/(\s+)/);
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    tokens.forEach((token) => {
      if (!token) return;
      if (/^\s+$/.test(token)) {
        fragment.appendChild(document.createTextNode(token));
        return;
      }
      if (cursor === attention.word) {
        fragment.appendChild(el("span", { class: "word-focus", text: token }));
      } else {
        fragment.appendChild(document.createTextNode(token));
      }
      cursor += 1;
    });
    caption.replaceChildren(fragment);
  }

  function updateAttentionImage() {
    const example = (DATA.attention_examples || [])[attention.index];
    if (!example) return;
    const overlay = $("#attention-overlay");
    const mode = $("#attention-toggle [data-mode][aria-pressed='true']")?.dataset.mode || "overlay";
    const record = (example.overlays || []).find((item) => item.index === attention.word);
    const status = $("#attention-status");
    if (!overlay) return;
    if (mode === "overlay" && record) {
      overlay.src = record.path;
      overlay.hidden = false;
      overlay.alt = `Attention overlay for “${record.word}”`;
      if (status) status.textContent = `Showing attention for “${record.word}”.`;
    } else {
      overlay.hidden = true;
      overlay.removeAttribute("src");
      if (status) {
        status.textContent =
          mode === "base"
            ? "Base image shown."
            : `No stored overlay for “${example.words[attention.word] || ""}”.`;
      }
    }
  }

  // ── Gallery ────────────────────────────────────────────────────────────
  const galleryState = { filter: "all", query: "" };

  function renderGallery() {
    const filters = $("#gallery-filters");
    if (filters) {
      clear(filters);
      [["all", "All models"], ...MODELS.map((id) => [id, label(id)])].forEach(([id, text]) => {
        const button = el("button", {
          class: "filter-chip",
          attrs: { type: "button", "aria-pressed": String(id === galleryState.filter) },
        }, [
          id !== "all" ? el("span", { class: "filter-chip__dot", style: { "--c": color(id) } }) : null,
          el("span", { text }),
        ].filter(Boolean));
        button.addEventListener("click", () => {
          galleryState.filter = id;
          $$("#gallery-filters .filter-chip").forEach((chip, index) => {
            chip.setAttribute("aria-pressed", String(["all", ...MODELS][index] === id));
          });
          renderGalleryItems();
        });
        filters.appendChild(button);
      });
    }

    const search = $("#gallery-search");
    if (search) {
      search.addEventListener("input", () => {
        galleryState.query = search.value.trim().toLowerCase();
        renderGalleryItems();
      });
    }
    renderGalleryItems();
  }

  function renderGalleryItems() {
    const root = $("#gallery-grid");
    if (!root) return;
    clear(root);
    const all = DATA.gallery || [];
    const filtered = all.filter((example) => {
      if (!galleryState.query) return true;
      const haystack = [
        String(example.image_id),
        ...(example.references || []),
        ...(example.gt_objects || []),
        ...MODELS.map((id) => example.captions?.[id]?.text || ""),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(galleryState.query);
    });
    filtered.forEach((example) => {
      const card = exampleCard(example);
      if (galleryState.filter !== "all") {
        $$(".caption-row", card).forEach((row) => {
          const marker = $(".caption-row__model", row)?.textContent || "";
          row.hidden = marker !== label(galleryState.filter);
        });
        const rows = $$(".caption-row", card).filter((row) => !row.hidden);
        rows.forEach((row) => row.classList.add("caption-row--selected"));
      }
      root.appendChild(card);
    });
    const empty = $("#gallery-empty");
    if (empty) {
      empty.hidden = filtered.length !== 0;
      empty.textContent = all.length
        ? "No examples match that filter and search."
        : "No gallery examples in this payload.";
    }
    const status = $("#gallery-status");
    if (status) status.textContent = `${filtered.length} of ${all.length} examples shown.`;
  }

  // ── Report ─────────────────────────────────────────────────────────────
  function renderReport() {
    const cards = $("#download-cards");
    if (cards) {
      clear(cards);
      (DATA.artifacts || []).forEach((artifact) => {
        const size = fileSize(artifact.size);
        cards.appendChild(
          el("a", { class: "download-card", attrs: { href: artifact.path, download: "" } }, [
            el("span", { class: "download-card__kind", text: String(artifact.kind || "file").toUpperCase() }),
            el("span", { class: "download-card__label", text: artifact.label }),
            el("span", { class: "download-card__meta" }, [
              el("span", { text: size || "download" }),
              el("span", { class: "download-card__arrow", text: "↓", attrs: { "aria-hidden": "true" } }),
            ]),
          ]),
        );
      });
    }

    const notes = $("#repro-notes");
    if (notes) {
      clear(notes);
      [
        `Seed ${DATA.meta.seed}; one training run per model.`,
        "Features and detector outputs are cached; the LSTM never reruns a CNN.",
        `Evaluation split: ${DATA.meta.split} (${compact(DATA.meta.n_test)} images).`,
        "Dashboard and report read the same saved metrics snapshot.",
        "Open this file locally; no network requests are made.",
      ].forEach((text) => notes.appendChild(el("li", { text })));
    }

    const figures = $("#figure-list");
    if (figures) {
      clear(figures);
      if (!(DATA.figures || []).length) {
        figures.appendChild(el("p", { class: "empty-note", text: "No figures in this payload." }));
      }
      (DATA.figures || []).forEach((figure) => {
        const button = el("button", {
          class: "figure-item",
          attrs: { type: "button", "data-lightbox-figure": figure.id },
        }, [
          el("img", { class: "figure-item__thumb", attrs: { src: figure.path, alt: "", loading: "lazy" } }),
          el("span", { class: "figure-item__label", text: figure.label }),
          el("span", { class: "figure-item__meta", text: "open" }),
        ]);
        figures.appendChild(button);
      });
    }

    const footer = $("#footer-meta");
    if (footer) footer.textContent = `${DATA.meta.title} · generated ${DATA.meta.generated_at || ""}`;
    wireFigureData();
  }

  // ── Figures ────────────────────────────────────────────────────────────
  function wireFigureData() {
    const map = new Map((DATA.figures || []).map((figure) => [figure.id, figure]));
    $$("[data-figure]").forEach((node) => {
      const figure = map.get(node.dataset.figure);
      if (!figure) {
        if (node.tagName === "IMG") {
          node.classList.add("img-missing");
          node.removeAttribute("src");
          node.setAttribute("alt", "Figure not available in this build");
        }
        const caption = node.parentElement?.querySelector("[data-figure-caption]");
        if (node.dataset.figureCaption !== undefined) node.textContent = "";
        else if (caption) caption.textContent = "Figure not available in this build.";
        return;
      }
      if (node.tagName === "IMG") {
        if (!node.getAttribute("src")) node.setAttribute("src", figure.path);
        node.addEventListener("error", () => node.classList.add("img-missing"));
      }
      if (node.dataset.figureCaption !== undefined) node.textContent = figure.caption || "";
    });
  }

  function wireFigureCards() {
    $$(".figure-card__zoom img, .figure-card__img img").forEach((image) => {
      const card = image.closest(".figure-card");
      const figureId = image.dataset.figure || card?.dataset.figure;
      if (!figureId) return;
      const trigger = card?.querySelector("[data-lightbox-target='figure']");
      image.addEventListener("click", () => trigger?.click());
    });
  }

  function wireLightbox() {
    const box = $("#lightbox");
    if (!box) return;
    let returnFocus = null;
    const open = (src, title, caption, alt) => {
      returnFocus = document.activeElement;
      const image = $("#lightbox-img");
      const error = $("#lightbox-error");
      image.hidden = false;
      if (error) error.hidden = true;
      image.src = src;
      image.alt = alt || "";
      image.onerror = () => {
        image.hidden = true;
        if (error) error.hidden = false;
      };
      $("#lightbox-title").textContent = title || "Figure";
      $("#lightbox-caption").textContent = caption || "";
      box.hidden = false;
      document.body.classList.add("is-locked");
      $(".lightbox__close", box)?.focus();
    };
    const close = () => {
      box.hidden = true;
      document.body.classList.remove("is-locked");
      if (returnFocus) returnFocus.focus();
    };
    box.addEventListener("click", (event) => {
      if (event.target.closest("[data-lightbox-close]")) close();
    });
    document.addEventListener("keydown", (event) => {
      if (box.hidden) return;
      if (event.key === "Escape") close();
      if (event.key === "Tab") {
        const focusables = $$("button, [href], [tabindex]:not([tabindex='-1'])", box).filter(
          (node) => !node.hidden,
        );
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    });

    document.addEventListener("click", (event) => {
      const trigger = event.target.closest("[data-lightbox-target]");
      if (trigger) {
        const type = trigger.dataset.lightboxTarget;
        if (type === "figure") {
          const id = trigger.dataset.figure;
          const figure = (DATA.figures || []).find((item) => item.id === id);
          if (!figure) return;
          event.preventDefault();
          open(figure.path, figure.label, figure.caption, figure.label);
        } else if (type === "attention") {
          const example = (DATA.attention_examples || [])[attention.index];
          if (!example) return;
          event.preventDefault();
          const mode = $("#attention-toggle [data-mode][aria-pressed='true']")?.dataset.mode;
          const record = (example.overlays || []).find((item) => item.index === attention.word);
          const src = mode === "base" || !record ? example.image : record.path;
          open(src, `Image ${example.image_id}`, example.caption, `Attention view for image ${example.image_id}`);
        }
        return;
      }
      const figureButton = event.target.closest("[data-lightbox-figure]");
      if (figureButton) {
        const figure = (DATA.figures || []).find((item) => item.id === figureButton.dataset.lightboxFigure);
        if (figure) open(figure.path, figure.label, figure.caption, figure.label);
      }
    });
    wireFigureCards();
  }

  function wireImageFallbacks() {
    $$("img").forEach((image) => {
      image.addEventListener("error", () => {
        image.classList.add("img-missing");
        image.setAttribute("aria-hidden", "true");
      });
    });
  }

  function wirePrint() {
    $$('[data-action="print"]').forEach((button) => {
      button.addEventListener("click", () => window.print());
    });
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
