// Adapt MyST's interactive-only nodes to constructs supported by static PDF
// renderers. Exercise conversion follows the official Jupyter Book
// exercise-admonition-pdf plugin, extended here to walk nested content and to
// preserve the book's existing algorithm, tab, grid, iframe, and anywidget
// content without changing the HTML build.

function isStaticDocumentBuild() {
  return process.argv.some((argument) =>
    ["pdf", "tex", "typst"].some((format) => argument.includes(format)),
  );
}

function walk(node, visitor) {
  if (!node || typeof node !== "object") return;
  visitor(node);
  if (Array.isArray(node.children)) {
    node.children.forEach((child) => walk(child, visitor));
  }
}

function hasNodeType(node, type) {
  let found = false;
  walk(node, (child) => {
    if (child.type === type) found = true;
  });
  return found;
}

function htmlOutputValue(node) {
  if (node?.type !== "output") return undefined;
  return node.jupyter_data?.data?.["text/html"];
}

function hasPortableStaticPayload(node) {
  const data = node?.jupyter_data?.data;
  if (!data || typeof data !== "object") return false;
  return [
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "application/pdf",
  ].some((mimeType) => data[mimeType]);
}

function staticPayloadSignature(node) {
  const data = node?.jupyter_data?.data;
  if (!data || typeof data !== "object") return undefined;

  for (const mimeType of [
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "application/pdf",
  ]) {
    const payload = data[mimeType];
    if (!payload) continue;
    if (typeof payload === "object") {
      return `${mimeType}:${payload.hash ?? payload.path ?? JSON.stringify(payload)}`;
    }
    return `${mimeType}:${String(payload)}`;
  }
  return undefined;
}

// MyST externalizes script-bearing HTML outputs for the web build. LaTeX has
// no representation for those browser applications and may otherwise emit an
// empty \includegraphics{} followed by fragments of Matplotlib's controls.
// Keep ordinary HTML tables (and any output with a real static image), while
// removing only browser-only output payloads from static exports.
function isBrowserOnlyHtmlOutput(node) {
  const html = htmlOutputValue(node);
  if (!html || hasPortableStaticPayload(node)) return false;

  if (typeof html === "object") {
    return html.content_type === "text/html" && Boolean(html.path || html.hash);
  }

  const value = Array.isArray(html) ? html.join("") : String(html);
  return /<script\b|<video\b|matplotlib[^<]{0,40}animation|class=["'][^"']*(?:animation|replay)/i.test(
    value,
  );
}

function pruneBrowserOnlyOutputs(node) {
  if (!node || typeof node !== "object" || !Array.isArray(node.children)) {
    return;
  }

  const children = [];
  node.children.forEach((child) => {
    if (isBrowserOnlyHtmlOutput(child)) return;
    if (child?.type === "image" && !child.url && !child.src) return;

    pruneBrowserOnlyOutputs(child);
    if (
      (child?.type === "outputs" || child?.type === "legend") &&
      Array.isArray(child.children) &&
      child.children.length === 0
    ) {
      return;
    }
    children.push(child);
  });

  if (node.type === "outputs") {
    const seenStaticPayloads = new Set();
    node.children = children.filter((child) => {
      const signature = staticPayloadSignature(child);
      if (!signature) return true;
      if (seenStaticPayloads.has(signature)) return false;
      seenStaticPayloads.add(signature);
      return true;
    });
    return;
  }
  node.children = children;
}

const staticCompanionSvgPatterns = [
  /^_static\/inference_serving\/(modeling|open-loop|mpc|scheduling|fqi)\.svg$/,
  /^_static\/bixi\/(bixi-model-interface|bixi-feedback-evidence|bixi-completed-trip-censoring)\.svg$/,
  /^_static\/gimbal\/partial-observability\.svg$/,
  /^_static\/swing_modeling\/model_audit\.svg$/,
  /^_static\/battery\/fast-charging\.svg$/,
  /^_static\/cubesat\/differential-drag\.svg$/,
];

// The HTML book keeps the responsive SVG fallbacks. For TeX, point the same
// image nodes at deterministic vector-PDF companions generated from those
// SVGs. This avoids MyST's ImageMagick fallback, whose build lacks FreeType
// and consequently drops the figures' text.
function usePdfCompanion(node) {
  if (node?.type !== "image") return;
  const source = node.urlSource ?? node.url;
  if (
    typeof source !== "string" ||
    !staticCompanionSvgPatterns.some((pattern) => pattern.test(source))
  ) {
    return;
  }

  const pdfSource = source.replace(/\.svg$/, ".pdf");
  node.urlSource = pdfSource;
  node.url = pdfSource;
}

function extractOutputGroups(node, extracted) {
  if (!node || typeof node !== "object" || !Array.isArray(node.children)) {
    return;
  }

  const children = [];
  node.children.forEach((child) => {
    if (child?.type === "outputs" || child?.type === "output") {
      extracted.push(child);
      return;
    }
    extractOutputGroups(child, extracted);
    children.push(child);
  });
  node.children = children;
}

// Executed figure cells currently arrive as code + caption + legend(outputs).
// The TeX renderer treats a legend as part of \caption{}, so block outputs
// placed there create invalid nested verbatim and graphics commands. Lift the
// output groups into the figure body and keep the prose caption last.
function normalizeExecutableFigure(node) {
  if (node?.type !== "container" || node.kind !== "figure") return;
  if (!Array.isArray(node.children)) return;

  const outputGroups = [];
  const body = [];
  const captions = [];

  node.children.forEach((child) => {
    if (child?.type === "caption" || child?.type === "legend") {
      extractOutputGroups(child, outputGroups);
      if (child.type === "caption") {
        captions.push(child);
      } else if (Array.isArray(child.children) && child.children.length > 0) {
        body.push(child);
      }
      return;
    }
    body.push(child);
  });

  if (outputGroups.length === 0) return;
  node.children = [...body, ...outputGroups, ...captions];
}

function nodeText(node) {
  if (!node || typeof node !== "object") return "";
  if (typeof node.value === "string") return node.value;
  if (!Array.isArray(node.children)) return "";
  return node.children.map(nodeText).join("").trim();
}

function titleNode(node) {
  return Array.isArray(node.children)
    ? node.children.find((child) => child.type === "admonitionTitle")
    : undefined;
}

function setAdmonitionTitle(node, value) {
  let title = titleNode(node);
  if (!title) {
    title = { type: "admonitionTitle", children: [] };
    node.children = [title, ...(node.children ?? [])];
  }
  title.children = [{ type: "text", value }];
}

function insertTexLabel(node, label, counter, number) {
  if (!label || !Array.isArray(node.children)) return;

  const parsedNumber = Number.parseInt(String(number), 10);
  const previousNumber = Number.isFinite(parsedNumber)
    ? Math.max(0, parsedNumber - 1)
    : 0;
  const counterSetup = counter
    ? `\\ifcsname c@${counter}\\endcsname\\else` +
      `\\newcounter{${counter}}\\fi` +
      `\\setcounter{${counter}}{${previousNumber}}` +
      `\\refstepcounter{${counter}}`
    : "";
  const rawLabel = {
    type: "raw",
    lang: "tex",
    tex: `\n${counterSetup}\\label{${label}}\n`,
  };
  const titleIndex = node.children.findIndex(
    (child) => child.type === "admonitionTitle",
  );
  node.children.splice(titleIndex + 1, 0, rawLabel);
}

function linkedExerciseLabel(node) {
  let label;
  walk(titleNode(node), (child) => {
    if (!label && child.type === "crossReference" && child.label) {
      label = child.label;
    }
  });
  return label;
}

// These figures contain recorded browser controls. Their sections also contain
// static analytical figures generated from the same Python trajectories.
// Static exports should use those figures rather than ask the LaTeX converter
// to interpret HTML or Matplotlib's animation player.
const browserReplayFigures = new Set([
  "fig-swing-structured-animation",
  "fig-bixi-control-replay",
  "fig-gimbal-observation-replay",
  "anim-cartpole-formulations",
  "fig-crane-collocation-animation",
  "fig-wave-economic-mpc-animation",
  "anim-cartpole-lqr",
  "fig-swing-ppo-replay",
  "fig-inference-serving-model",
  "fig-inference-open-loop",
  "fig-inference-mpc",
  "fig-inference-scheduling-dp",
  "fig-inference-scheduling-fqi",
]);

const pdfStaticParityTransform = {
  name: "pdf-static-parity",
  doc: "Convert interactive and rich-layout nodes for static document builds.",
  stage: "document",
  plugin: () => (tree) => {
    if (!isStaticDocumentBuild()) return;

    const exercises = new Map();
    let algorithmCount = 0;

    pruneBrowserOnlyOutputs(tree);
    walk(tree, usePdfCompanion);

    walk(tree, (node) => {
      if (node.type !== "exercise") return;

      const number = String(node.enumerator ?? exercises.size + 1);
      const originalTitle = nodeText(titleNode(node)) || "Self-check";
      if (node.label) exercises.set(node.label, { number, title: originalTitle });

      node.type = "admonition";
      node.kind = "note";
      delete node.class;
      setAdmonitionTitle(node, `Exercise ${number}: ${originalTitle}`);
      insertTexLabel(
        node,
        node.identifier ?? node.label,
        "rlexercise",
        number,
      );
    });

    walk(tree, (node) => {
      if (
        node.type === "container" &&
        node.kind === "figure" &&
        browserReplayFigures.has(node.identifier ?? node.label)
      ) {
        node.type = "block";
        node.children = [];
        delete node.kind;
        delete node.label;
        delete node.identifier;
        delete node.enumerator;
        return;
      }

      if (
        node.type === "container" &&
        node.kind === "figure" &&
        (hasNodeType(node, "output") || hasNodeType(node, "outputs"))
      ) {
        normalizeExecutableFigure(node);
      }

      if (
        node.type === "crossReference" &&
        (node.kind === "proof:algorithm" ||
          node.kind === "prf:ref" ||
          nodeText(node).startsWith("Algorithm"))
      ) {
        node.children = [{ type: "text", value: "%s" }];
        node.template = "Algorithm %s";
        return;
      }

      if (node.type === "solution") {
        const linked = exercises.get(linkedExerciseLabel(node));
        const suffix = linked?.title ? `: ${linked.title}` : "";
        const number = linked?.number ? ` ${linked.number}` : "";

        node.type = "admonition";
        node.kind = "tip";
        delete node.class;
        setAdmonitionTitle(node, `Solution to Exercise${number}${suffix}`);
        return;
      }

      if (node.type === "proof" && node.kind === "algorithm") {
        algorithmCount += 1;
        const number = String(node.enumerator ?? algorithmCount);
        const originalTitle = nodeText(titleNode(node)) || "Procedure";
        node.type = "admonition";
        node.kind = "note";
        setAdmonitionTitle(node, `Algorithm ${number}: ${originalTitle}`);
        insertTexLabel(
          node,
          node.identifier ?? node.label,
          "rlalgorithm",
          number,
        );
        return;
      }

      // myst-to-tex labels ordinary display math, but not an explicit top-level
      // align environment. Nest a labelled align in the generated equation so
      // the PDF cross-reference remains resolvable and receives one number.
      if (
        node.type === "math" &&
        (node.identifier || node.label) &&
        /^\s*\\begin\{align\*\}/.test(node.value ?? "")
      ) {
        node.value = node.value
          .replace(/^\s*\\begin\{align\*\}/, "\\begin{aligned}")
          .replace(/\\end\{align\*\}\s*$/, "\\end{aligned}");
        return;
      }

      if (node.type === "tabSet" || node.type === "grid" || node.type === "grid-item") {
        node.type = "block";
        return;
      }

      if (node.type === "tabItem") {
        const heading = node.title
          ? {
              type: "heading",
              depth: 1,
              children: [{ type: "text", value: node.title }],
            }
          : undefined;
        node.type = "block";
        if (heading) node.children = [heading, ...(node.children ?? [])];
        return;
      }

      // Disclosure widgets in this book contain expandable implementation
      // source. They have no useful static analogue, and the TeX exporter
      // otherwise reports every <details> node as an unhandled conversion.
      if (node.type === "details") {
        node.type = "block";
        node.children = [];
        return;
      }

      if (node.type === "iframe" || node.type === "anywidget") {
        node.type = "block";
        node.children = [];
      }
    });
  },
};

export default {
  name: "PDF static parity",
  transforms: [pdfStaticParityTransform],
};
