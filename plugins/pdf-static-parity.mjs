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
    ? `\\setcounter{${counter}}{${previousNumber}}` +
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

const pdfStaticParityTransform = {
  name: "pdf-static-parity",
  doc: "Convert interactive and rich-layout nodes for static document builds.",
  stage: "document",
  plugin: () => (tree) => {
    if (!isStaticDocumentBuild()) return;

    const exercises = new Map();
    let algorithmCount = 0;

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
