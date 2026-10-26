/**
 * Build the college presentation on the NMIMS template:
 *   presentation/AskMyPDF_NMIMS_Presentation.pptx   (10 slides)
 *
 *     npm install pptxgenjs      (once)
 *     node scripts/make_college_ppt.js
 *
 * The red banner, the logo box and the bottom bar copy the department
 * template; the two logos are the real ones lifted from it.
 *
 * DEV TOOL -- not needed to use AskMyPDF.
 */

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT_DIR = path.join(ROOT, "presentation");
const OUT = path.join(OUT_DIR, "AskMyPDF_NMIMS_Presentation.pptx");
const ASSETS = path.join(ROOT, "presentation", "assets");
const LOGO_NMIMS = path.join(ASSETS, "nmims_logo.jpeg");
const LOGO_SCHOOL = path.join(ASSETS, "school_logo.png");
const SCREENSHOT = path.join(ROOT, "docs", "screenshot.png");

// --------------------------------------------------------------------------
// Palette taken from the template: university red on white, with grey text.
// --------------------------------------------------------------------------
const RED = "C00000";       // banner + headings
const DARK_RED = "8B0000";  // secondary accent
const INK = "1A1A1A";       // body text
const GREY = "595959";      // captions
const PANEL = "F4F4F4";     // card fill
const LINE = "D9D9D9";      // card border
const WHITE = "FFFFFF";

const TITLE_FONT = "Cambria";   // matches the template's serif headings
const BODY_FONT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";    // 13.33 x 7.5in, same as the template
pres.theme = { headFontFace: TITLE_FONT, bodyFontFace: BODY_FONT };
pres.title = "AskMyPDF - Extractive Question Answering over PDFs";
pres.company = "NMIMS Hyderabad";

// --------------------------------------------------------------------------
// The template frame, drawn on every slide
// --------------------------------------------------------------------------
const BANNER_H = 0.86;
const BODY_TOP = 1.72;

function frame(slide, headingText) {
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: BANNER_H, fill: { color: RED },
    line: { color: RED, width: 0 }, objectName: "banner",
  });
  // White box holding the university logo, overlapping the banner.
  slide.addShape(pres.ShapeType.rect, {
    x: 0.08, y: 0.17, w: 2.62, h: 1.33, fill: { color: WHITE },
    line: { color: LINE, width: 1 }, objectName: "logo-box",
  });
  if (fs.existsSync(LOGO_NMIMS)) {
    slide.addImage({ path: LOGO_NMIMS, x: 0.22, y: 0.36, w: 2.32, h: 0.95 });
  }
  if (headingText) {
    slide.addText(headingText, {
      x: 2.85, y: 0.1, w: 10.2, h: 0.66, align: "center", valign: "middle",
      fontSize: 25, bold: true, color: WHITE, fontFace: TITLE_FONT,
      margin: 0, isTextBox: true,
    });
  }
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 7.3, w: 13.33, h: 0.2, fill: { color: RED },
    line: { color: RED, width: 0 }, objectName: "bottom-bar",
  });
}

function card(slide, { x, y, w, h, fill = PANEL, line = LINE, name }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06, fill: { color: fill },
    line: { color: line, width: 1 }, objectName: name,
  });
}

function badge(slide, { x, y, n, size = 0.4, color = RED }) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: size, h: size, fill: { color }, line: { color, width: 0 },
    objectName: `badge-${n}`,
  });
  slide.addText(String(n), {
    x, y, w: size, h: size, align: "center", valign: "middle",
    fontSize: 13, bold: true, color: WHITE, margin: 0, isTextBox: true,
  });
}

function sectionHead(slide, text, { x = 0.6, y = BODY_TOP, w = 12.1 } = {}) {
  slide.addText(text, {
    x, y, w, h: 0.42, fontSize: 19, bold: true, color: RED,
    fontFace: TITLE_FONT, margin: 0, isTextBox: true,
  });
}

function bullets(slide, items, opts) {
  slide.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: { code: "25AA" }, breakLine: i < items.length - 1 },
    })),
    Object.assign({ fontSize: 14, color: INK, paraSpaceAfter: 7, valign: "top",
                    isTextBox: true }, opts)
  );
}

function stat(slide, { x, y, w, value, label, color = RED, size = 34 }) {
  slide.addText(value, {
    x, y, w, h: 0.62, fontSize: size, bold: true, color, fontFace: TITLE_FONT,
    align: "center", margin: 0, isTextBox: true,
  });
  slide.addText(label, {
    x, y: y + 0.6, w, h: 0.5, fontSize: 11.5, color: GREY, align: "center",
    margin: 0, isTextBox: true,
  });
}

// ==========================================================================
// Slide 1 — Title (follows the department's title-slide layout)
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, null);

  s.addText("AskMyPDF", {
    x: 1.0, y: 2.35, w: 11.3, h: 0.95, align: "center", fontSize: 52, bold: true,
    color: RED, fontFace: TITLE_FONT, margin: 0, isTextBox: true,
  });
  s.addText("Extractive Question Answering over PDF Documents using Transformers", {
    x: 1.0, y: 3.38, w: 11.3, h: 0.45, align: "center", fontSize: 18, color: INK,
    margin: 0, isTextBox: true,
  });
  s.addText("Natural Language Processing — Mini Project", {
    x: 1.0, y: 3.95, w: 11.3, h: 0.4, align: "center", fontSize: 15, color: GREY,
    margin: 0, isTextBox: true,
  });
  s.addText("Hugging Face Transformers  ·  RoBERTa & XLM-RoBERTa  ·  Streamlit  ·  EasyOCR", {
    x: 1.0, y: 4.5, w: 11.3, h: 0.4, align: "center", fontSize: 12.5, color: DARK_RED,
    margin: 0, isTextBox: true,
  });

  // Bottom band: school logo, team, mentor
  if (fs.existsSync(LOGO_SCHOOL)) {
    s.addImage({ path: LOGO_SCHOOL, x: 0.5, y: 5.75, w: 3.9, h: 0.86 });
  }
  s.addText("HYDERABAD", {
    x: 1.86, y: 6.52, w: 2.6, h: 0.3, fontSize: 11, bold: true, color: "404040",
    margin: 0, isTextBox: true,
  });

  s.addText([
    { text: "Team Members", options: { bold: true, fontSize: 13, color: RED, breakLine: true } },
    { text: "G. Vikram  —  70572300022", options: { fontSize: 13, color: INK, breakLine: true } },
    { text: "B. Dhanush  —  70022300509", options: { fontSize: 13, color: INK } },
  ], { x: 5.2, y: 5.85, w: 3.7, h: 1.1, valign: "top", margin: 0, isTextBox: true,
       paraSpaceAfter: 3 });

  s.addText([
    { text: "Mentor", options: { bold: true, fontSize: 13, color: RED, breakLine: true } },
    { text: "Niladri Sath", options: { fontSize: 13, color: INK } },
  ], { x: 9.4, y: 5.85, w: 3.4, h: 1.1, valign: "top", margin: 0, isTextBox: true,
       paraSpaceAfter: 3 });

  s.addNotes("Introduce the project in one sentence: ask a question about any PDF and get the " +
    "exact answer back, with the page it came from. Stress 'extractive' — the model copies text " +
    "from the document, it does not generate it.");
}

// ==========================================================================
// Slide 2 — Agenda
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Agenda / Contents");

  const items = [
    ["Problem Statement", "why searching a PDF by hand fails"],
    ["Objectives", "what the system must do"],
    ["System Architecture", "PDF → text → chunks → model → answer"],
    ["NLP Concepts Applied", "tokenization, chunking, span prediction"],
    ["Models & Hugging Face", "RoBERTa and XLM-RoBERTa on SQuAD 2.0"],
    ["Implementation", "modules, interface, technology stack"],
    ["Multilingual Support & OCR", "~100 languages, scanned books"],
    ["Results, Limitations, Conclusion", "what works and what does not"],
  ];
  let y = BODY_TOP + 0.05;
  items.forEach(([head, sub], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.25;
    const yy = y + row * 1.22;
    card(s, { x, y: yy, w: 5.95, h: 1.05, name: `agenda-${i}` });
    badge(s, { x: x + 0.25, y: yy + 0.3, n: i + 1, size: 0.44 });
    s.addText(head, {
      x: x + 0.85, y: yy + 0.17, w: 4.9, h: 0.38, fontSize: 15, bold: true,
      color: INK, margin: 0, isTextBox: true,
    });
    s.addText(sub, {
      x: x + 0.85, y: yy + 0.55, w: 4.9, h: 0.35, fontSize: 11.5, color: GREY,
      margin: 0, isTextBox: true,
    });
  });

  s.addNotes("Keep this short — read the eight headings and move on.");
}

// ==========================================================================
// Slide 3 — Problem statement and objectives
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Problem Statement & Objectives");

  sectionHead(s, "The problem", { w: 6.0 });
  bullets(s, [
    "Finding one fact in a long document means reading all of it",
    "Ctrl+F only works if you guess the author's exact words",
    "Scanned books hold no searchable text at all",
    "Indian-language and other non-English PDFs are served worst",
  ], { x: 0.6, y: BODY_TOP + 0.5, w: 6.0, h: 2.0 });

  sectionHead(s, "Objectives", { y: BODY_TOP + 2.55, w: 6.0 });
  bullets(s, [
    "Answer a plain-language question from any PDF",
    "Show confidence and the exact source passage",
    "Handle long documents, many languages and scanned pages",
    "Say “no answer” instead of guessing",
  ], { x: 0.6, y: BODY_TOP + 3.05, w: 6.0, h: 2.0 });

  // Example panel
  card(s, { x: 7.0, y: BODY_TOP, w: 5.7, h: 5.05, name: "example" });
  s.addText("WORKED EXAMPLE", {
    x: 7.35, y: BODY_TOP + 0.25, w: 5.0, h: 0.3, fontSize: 11, bold: true,
    color: GREY, charSpacing: 1, margin: 0, isTextBox: true,
  });
  s.addText("“Which computer defeated Garry Kasparov in 1997?”", {
    x: 7.35, y: BODY_TOP + 0.62, w: 5.0, h: 0.6, fontSize: 15, italic: true,
    color: INK, margin: 0, isTextBox: true,
  });
  s.addText("Document: 3 pages, 1,354 words", {
    x: 7.35, y: BODY_TOP + 1.35, w: 5.0, h: 0.3, fontSize: 12, color: GREY,
    margin: 0, isTextBox: true,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.35, y: BODY_TOP + 1.8, w: 2.6, h: 0.62, rectRadius: 0.08,
    fill: { color: RED }, line: { color: RED, width: 0 }, objectName: "answer-chip",
  });
  s.addText("Deep Blue", {
    x: 7.35, y: BODY_TOP + 1.8, w: 2.6, h: 0.62, align: "center", valign: "middle",
    fontSize: 20, bold: true, color: WHITE, margin: 0, isTextBox: true,
  });
  s.addText([
    { text: "Confidence:  ", options: { color: GREY, fontSize: 13 } },
    { text: "76.7%", options: { color: INK, fontSize: 13, bold: true, breakLine: true } },
    { text: "Source:  ", options: { color: GREY, fontSize: 13 } },
    { text: "chunk 3 of 4, pages 2–3", options: { color: INK, fontSize: 13, bold: true } },
  ], { x: 7.35, y: BODY_TOP + 2.65, w: 5.0, h: 0.9, margin: 0, isTextBox: true, paraSpaceAfter: 4 });
  s.addText("The answer is copied from the document — the model cannot invent a fact, " +
    "and it returns nothing when the document does not contain one.", {
    x: 7.35, y: BODY_TOP + 3.75, w: 5.0, h: 1.0, fontSize: 12, color: GREY,
    italic: true, margin: 0, isTextBox: true,
  });

  s.addNotes("The 'no answer' behaviour is deliberate: the models are trained on SQuAD 2.0, " +
    "which contains unanswerable questions on purpose.");
}

// ==========================================================================
// Slide 4 — System architecture
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "System Architecture");

  const steps = [
    ["PDF input", "Uploaded in the browser"],
    ["Text extraction", "pdfplumber, PyPDF2 fallback"],
    ["Scanned check", "If image → OCR"],
    ["Chunking", "Overlapping, token-sized"],
    ["QA model", "Each chunk read with the question"],
    ["Best answer", "Ranked, with its source"],
  ];
  // Start below the logo box so the first badge clears it.
  const ARCH_TOP = BODY_TOP + 0.34;
  const w = 1.92, gap = 0.17;
  let x = 0.62;
  steps.forEach(([h, b], i) => {
    card(s, { x, y: ARCH_TOP, w, h: 1.9, name: `arch-${i}` });
    badge(s, { x: x + w / 2 - 0.2, y: ARCH_TOP - 0.2, n: i + 1 });
    s.addText(h, {
      x: x + 0.1, y: ARCH_TOP + 0.45, w: w - 0.2, h: 0.4, align: "center",
      fontSize: 13.5, bold: true, color: INK, margin: 0, isTextBox: true,
    });
    s.addText(b, {
      x: x + 0.1, y: ARCH_TOP + 0.9, w: w - 0.2, h: 0.85, align: "center",
      fontSize: 11, color: GREY, margin: 0, isTextBox: true,
    });
    if (i < steps.length - 1) {
      s.addText("➤", {
        x: x + w, y: ARCH_TOP + 0.78, w: gap, h: 0.3, align: "center",
        fontSize: 11, color: RED, margin: 0, isTextBox: true,
      });
    }
    x += w + gap;
  });

  card(s, { x: 0.62, y: ARCH_TOP + 2.25, w: 6.0, h: 1.6, fill: "FDF2F2",
    line: "EFC9C9", name: "note-chunk" });
  s.addText("Why chunking is necessary", {
    x: 0.92, y: ARCH_TOP + 2.45, w: 5.4, h: 0.35, fontSize: 14.5, bold: true,
    color: DARK_RED, margin: 0, isTextBox: true,
  });
  s.addText("A transformer reads at most 512 tokens at once — about 400 English words. " +
    "A 10,000-word document is cut into overlapping chunks, and every chunk is asked the " +
    "same question.", {
    x: 0.92, y: ARCH_TOP + 2.85, w: 5.4, h: 0.9, fontSize: 12, color: INK,
    margin: 0, isTextBox: true,
  });

  card(s, { x: 6.9, y: ARCH_TOP + 2.25, w: 5.8, h: 1.6, name: "note-rank" });
  s.addText("How the best chunk is chosen", {
    x: 7.2, y: ARCH_TOP + 2.45, w: 5.2, h: 0.35, fontSize: 14.5, bold: true,
    color: DARK_RED, margin: 0, isTextBox: true,
  });
  s.addText("Each chunk also scores “no answer here”. Ranking by (answer score − no-answer " +
    "score) makes chunks comparable — a raw score only means something inside its own chunk.", {
    x: 7.2, y: ARCH_TOP + 2.85, w: 5.2, h: 0.9, fontSize: 12, color: INK,
    margin: 0, isTextBox: true,
  });

  s.addText("Modules:  pdf_extractor.py  →  ocr.py  →  chunker.py  →  qa_engine.py  →  app.py", {
    x: 0.62, y: ARCH_TOP + 4.1, w: 12.1, h: 0.4, fontSize: 12.5, color: GREY,
    italic: true, margin: 0, isTextBox: true,
  });

  s.addNotes("This is the main technical slide — walk left to right, then explain the two notes.");
}

// ==========================================================================
// Slide 5 — NLP concepts
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "NLP Concepts Applied");

  const concepts = [
    ["Extractive question answering", "The model predicts where the answer starts and ends, " +
      "and returns that span — it never generates text."],
    ["Sub-word tokenization", "BPE and SentencePiece split words into pieces. Measured: ~1.2 " +
      "tokens per word in English, ~2.7 in Telugu."],
    ["Context window & chunking", "512 tokens maximum, so documents are split into overlapping " +
      "windows sized from the measured token ratio."],
    ["Overlap between chunks", "Chunks share 50 words so an answer on a boundary stays whole " +
      "in at least one of them."],
    ["Unanswerable questions", "Trained on SQuAD 2.0, the model can decline; that null score " +
      "also makes chunks comparable."],
    ["Cross-lingual transfer", "Fine-tuned for QA in English only, yet answers in ~100 " +
      "languages — zero-shot transfer."],
  ];
  let cy = BODY_TOP;
  concepts.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.62 + col * 6.28;
    const y = cy + row * 1.72;
    card(s, { x, y, w: 5.98, h: 1.55, name: `nlp-${i}` });
    s.addText(h, {
      x: x + 0.3, y: y + 0.18, w: 5.4, h: 0.38, fontSize: 14.5, bold: true,
      color: RED, margin: 0, isTextBox: true,
    });
    s.addText(b, {
      x: x + 0.3, y: y + 0.58, w: 5.4, h: 0.85, fontSize: 12, color: INK,
      margin: 0, isTextBox: true,
    });
  });

  s.addNotes("Examiners usually ask 'where exactly is the NLP?' — this slide is the answer.");
}

// ==========================================================================
// Slide 6 — Models and Hugging Face integration
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Models & Hugging Face Integration");

  const models = [
    { x: 0.62, role: "Default — English", name: "deepset/roberta-base-squad2",
      rows: [["Parameters", "124 M"], ["Languages", "English"],
             ["SQuAD 2.0 score", "82.95 F1"], ["Speed", "≈ 0.2 s per chunk"]] },
    { x: 6.95, role: "Multilingual", name: "deepset/xlm-roberta-large-squad2",
      rows: [["Parameters", "560 M"], ["Languages", "~100"],
             ["Fine-tuned on", "SQuAD 2.0 (English)"], ["Size on disk", "≈ 2.2 GB"]] },
  ];
  models.forEach((m, i) => {
    card(s, { x: m.x, y: BODY_TOP, w: 5.76, h: 2.65, name: `model-${i}` });
    s.addText(m.role, {
      x: m.x + 0.3, y: BODY_TOP + 0.18, w: 5.1, h: 0.3, fontSize: 11, bold: true,
      color: RED, charSpacing: 1, margin: 0, isTextBox: true,
    });
    s.addText(m.name, {
      x: m.x + 0.3, y: BODY_TOP + 0.5, w: 5.2, h: 0.35, fontSize: 13.5, bold: true,
      color: INK, fontFace: "Consolas", margin: 0, isTextBox: true,
    });
    m.rows.forEach(([k, v], j) => {
      const y = BODY_TOP + 0.98 + j * 0.4;
      s.addText(k, { x: m.x + 0.3, y, w: 2.9, h: 0.33, fontSize: 12, color: GREY,
        margin: 0, isTextBox: true });
      s.addText(v, { x: m.x + 3.2, y, w: 2.2, h: 0.33, fontSize: 12, bold: true,
        color: INK, align: "right", margin: 0, isTextBox: true });
    });
  });

  sectionHead(s, "How the library is used", { y: BODY_TOP + 2.95 });
  card(s, { x: 0.62, y: BODY_TOP + 3.4, w: 5.76, h: 1.0, fill: "F8F8F8", name: "code" });
  s.addText([
    { text: "from transformers import pipeline", options: { breakLine: true } },
    { text: "qa = pipeline(\"question-answering\",", options: { breakLine: true } },
    { text: "              model=\"deepset/roberta-base-squad2\")", options: {} },
  ], { x: 0.85, y: BODY_TOP + 3.55, w: 5.3, h: 0.75, fontSize: 11,
       fontFace: "Consolas", color: INK, margin: 0, isTextBox: true });

  bullets(s, [
    "Model details fetched live from the Hugging Face Hub API and shown in the app",
    "Runs locally, or on Hugging Face's hosted Inference API with a token",
    "RoBERTa chosen over BERT: same size, pre-trained on ~10× more text",
  ], { x: 6.95, y: BODY_TOP + 3.4, w: 5.76, h: 1.2, fontSize: 12.5 });

  s.addNotes("If asked why not a large language model: this is extractive, runs offline, " +
    "cites its source and cannot hallucinate.");
}

// ==========================================================================
// Slide 7 — Implementation
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Implementation & Technology Stack");

  if (fs.existsSync(SCREENSHOT)) {
    s.addImage({ path: SCREENSHOT, x: 0.62, y: BODY_TOP, w: 7.3, h: 4.44 });
  }
  s.addText("Streamlit web interface", {
    x: 0.62, y: BODY_TOP + 4.5, w: 7.3, h: 0.3, fontSize: 11.5, color: GREY,
    align: "center", italic: true, margin: 0, isTextBox: true,
  });

  sectionHead(s, "Modules", { x: 8.2, w: 4.5 });
  const modules = [
    ["pdf_extractor.py", "PDF → text, column handling, repairs"],
    ["ocr.py", "scanned pages → text with EasyOCR"],
    ["chunker.py", "overlapping, token-sized chunks"],
    ["qa_engine.py", "the model, scoring, selection"],
    ["app.py", "interface only, no NLP logic"],
  ];
  let my = BODY_TOP + 0.52;
  modules.forEach(([f, d], i) => {
    s.addText(f, { x: 8.2, y: my, w: 2.3, h: 0.44, fontSize: 11, bold: true,
      color: RED, fontFace: "Consolas", valign: "top", margin: 0, isTextBox: true });
    s.addText(d, { x: 10.5, y: my, w: 2.3, h: 0.44, fontSize: 11, color: INK,
      valign: "top", margin: 0, isTextBox: true });
    my += 0.52;
  });

  sectionHead(s, "Technology stack", { x: 8.2, y: BODY_TOP + 3.2, w: 4.5 });
  s.addText("Python 3.10 · Hugging Face transformers · PyTorch (CUDA) · " +
    "Streamlit · pdfplumber · PyPDF2 · EasyOCR · pypdfium2", {
    x: 8.2, y: BODY_TOP + 3.65, w: 4.6, h: 1.0, fontSize: 12, color: INK,
    margin: 0, isTextBox: true,
  });

  s.addNotes("Demo here if the laptop allows: pick the sample PDF, ask the Kasparov question, " +
    "show the highlighted source passage.");
}

// ==========================================================================
// Slide 8 — Multilingual + OCR
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Multilingual Support & OCR");

  stat(s, { x: 0.62, y: BODY_TOP, w: 2.9, value: "26", label: "languages tested" });
  stat(s, { x: 3.72, y: BODY_TOP, w: 2.9, value: "45/55", label: "questions correct" });
  stat(s, { x: 6.82, y: BODY_TOP, w: 2.9, value: "25/25", label: "Indian-language questions" });
  stat(s, { x: 9.92, y: BODY_TOP, w: 2.8, value: "15/22", label: "scheduled Indian languages" });

  s.addText("100% correct: English, French, Spanish, German, Portuguese, Russian, Turkish, " +
    "Chinese, Japanese, Korean, Arabic, Urdu, Sindhi, Hindi, Tamil, Punjabi", {
    x: 0.62, y: BODY_TOP + 1.35, w: 12.1, h: 0.55, fontSize: 12.5, color: INK,
    margin: 0, isTextBox: true,
  });

  card(s, { x: 0.62, y: BODY_TOP + 2.0, w: 6.0, h: 1.95, fill: "FDF2F2",
    line: "EFC9C9", name: "indic" });
  s.addText("Where it struggles — and why", {
    x: 0.92, y: BODY_TOP + 2.18, w: 5.4, h: 0.35, fontSize: 14.5, bold: true,
    color: DARK_RED, margin: 0, isTextBox: true,
  });
  s.addText("In Telugu, Kannada and Bengali one question failed each — the one whose answer " +
    "holds a conjunct letter. The PDF stores it as a single glyph with no mapping back to " +
    "Unicode, so ఆగ్రా arrives as ఆ. It is a PDF limitation, not a model limitation.", {
    x: 0.92, y: BODY_TOP + 2.58, w: 5.4, h: 1.25, fontSize: 11.5, color: INK,
    margin: 0, isTextBox: true,
  });

  card(s, { x: 6.9, y: BODY_TOP + 2.0, w: 5.8, h: 1.95, name: "ocr" });
  s.addText("OCR reads what the text layer cannot", {
    x: 7.2, y: BODY_TOP + 2.18, w: 5.2, h: 0.35, fontSize: 14.5, bold: true,
    color: DARK_RED, margin: 0, isTextBox: true,
  });
  s.addText("A 39-page scanned Telugu book held 10 words of text — a watermark. Rendering the " +
    "pages as images and running EasyOCR recovered 2,681 words, and the questions were answered " +
    "in Telugu and in English. Telugu, Kannada and Bengali go 1/2 → 2/2.", {
    x: 7.2, y: BODY_TOP + 2.58, w: 5.2, h: 1.25, fontSize: 11.5, color: INK,
    margin: 0, isTextBox: true,
  });

  s.addNotes("Best story in the project: a failure that looks like the model's fault but is " +
    "really the file format's, diagnosed and worked around.");
}

// ==========================================================================
// Slide 9 — Results, testing, limitations
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Results, Testing & Limitations");

  sectionHead(s, "Testing and results", { w: 6.0 });
  const rows = [
    [{ text: "What was tested", options: { bold: true } },
     { text: "Result", options: { bold: true, align: "center" } }],
    ["Automated tests (unit + end-to-end + UI)", { text: "48 pass", options: { align: "center" } }],
    ["English sample questions", { text: "7 / 7", options: { align: "center" } }],
    ["Indian-language questions (clean text)", { text: "25 / 25", options: { align: "center" } }],
    ["26-language PDF test set", { text: "45 / 55", options: { align: "center" } }],
    ["Dense research papers", { text: "≈ 4 / 10", options: { align: "center" } }],
  ];
  s.addTable(rows, {
    x: 0.62, y: BODY_TOP + 0.5, w: 6.0, colW: [4.2, 1.8], rowH: 0.42,
    fontSize: 12, color: INK, border: { type: "solid", color: LINE, pt: 1 },
    fill: { color: WHITE }, valign: "middle", margin: 5,
  });
  s.addText("Tests cover chunk overlap, the extractor fallback, scanned-PDF detection, " +
    "every error path, and questions that must return “no answer”.", {
    x: 0.62, y: BODY_TOP + 3.3, w: 6.0, h: 0.7, fontSize: 11.5, color: GREY,
    italic: true, margin: 0, isTextBox: true,
  });

  sectionHead(s, "Known limitations", { x: 6.95, w: 5.8 });
  bullets(s, [
    "Extractive only — no summarising, counting or arithmetic",
    "Reads one chunk at a time, so facts spread across distant pages cannot be joined",
    "Tables and complex layouts arrive as flat rows of digits",
    "Answers stay in the document's language (no translation step)",
    "No OCR model for Gujarati, Malayalam, Odia or Punjabi",
    "Multilingual model needs ~2.5 GB free memory; OCR ~20 s per page on CPU",
  ], { x: 6.95, y: BODY_TOP + 0.5, w: 5.8, h: 3.4, fontSize: 12.5 });

  s.addNotes("Do not skip the limitations — stating precisely where the system fails, with " +
    "numbers, is what separates a tested project from a demo.");
}

// ==========================================================================
// Slide 10 — Conclusion and future work
// ==========================================================================
{
  const s = pres.addSlide();
  frame(s, "Conclusion & Future Work");

  sectionHead(s, "Conclusion");
  s.addText("A working question-answering system over PDF documents: it locates the answer " +
    "inside the document, reports how confident it is, and shows the page it came from — " +
    "across ~100 languages and on scanned books.", {
    x: 0.62, y: BODY_TOP + 0.48, w: 12.1, h: 0.75, fontSize: 14, color: INK,
    margin: 0, isTextBox: true,
  });

  const takeaways = [
    ["Chunking shapes everything", "The 512-token limit drives every design decision in the pipeline"],
    ["Extraction is half the battle", "Most wrong answers came from bad text, not from a bad model"],
    ["Confidence must be earned", "Comparing chunks needs the margin over saying nothing"],
  ];
  let tx = 0.62;
  takeaways.forEach(([h, b], i) => {
    card(s, { x: tx, y: BODY_TOP + 1.35, w: 3.92, h: 1.45, name: `take-${i}` });
    s.addText(h, { x: tx + 0.28, y: BODY_TOP + 1.55, w: 3.4, h: 0.4, fontSize: 14,
      bold: true, color: RED, margin: 0, isTextBox: true });
    s.addText(b, { x: tx + 0.28, y: BODY_TOP + 1.95, w: 3.4, h: 0.75, fontSize: 11.5,
      color: INK, margin: 0, isTextBox: true });
    tx += 4.1;
  });

  sectionHead(s, "Future work", { y: BODY_TOP + 3.05 });
  bullets(s, [
    "Translate the extracted answer into the question's language",
    "Batch chunks on the GPU for faster answering on long documents",
    "Add Tesseract OCR for Gujarati, Malayalam, Odia, Punjabi and Tamil",
    "Table-aware extraction so numeric questions work on research papers",
  ], { x: 0.62, y: BODY_TOP + 3.5, w: 7.6, h: 1.6, fontSize: 12.5 });

  card(s, { x: 8.6, y: BODY_TOP + 3.5, w: 4.12, h: 1.35, fill: "FDF2F2",
    line: "EFC9C9", name: "repo" });
  s.addText("Project repository", {
    x: 8.85, y: BODY_TOP + 3.68, w: 3.7, h: 0.3, fontSize: 12, bold: true,
    color: DARK_RED, margin: 0, isTextBox: true,
  });
  s.addText("github.com/Vikram5002/AskMyPDF", {
    x: 8.85, y: BODY_TOP + 4.0, w: 3.7, h: 0.6, fontSize: 11.5, color: INK,
    fontFace: "Consolas", margin: 0, isTextBox: true,
  });

  s.addNotes("Close on the three takeaways, give the repository link, and invite questions.");
}

fs.mkdirSync(OUT_DIR, { recursive: true });
pres.writeFile({ fileName: OUT }).then(() => console.log("Wrote " + OUT));
