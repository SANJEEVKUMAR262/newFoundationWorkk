#!/usr/bin/env python3
"""
FULL LOCAL PIPELINE — All stages from the Colab notebook, adapted for local execution.
======================================================================================

STAGES:
  1. Delimiter Extraction    — Parse .mmd → extract Q&A sections
  2. LLM Question Pipeline   — Extract → Route → Generate Options → Solve → Difficulty
  3. Cleaning                — Strip OCR noise, normalize formatting
  4. Dedup (Embedding)       — Create embeddings, find near-duplicates, drop them
  5. H3 Tagging + Metadata   — Wrap in <h3>, add class/mode/source flags
  6. MathML Conversion       — Convert LaTeX → MathML
  7. Concept Tagging          — Use embeddings + GPT to tag chapter/concept/subConcept

OUTPUT: Final JSON ready for upload (same schema as Acadza DB)
"""

from __future__ import annotations
import json, logging, re, time, hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel
from openai import OpenAI
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ============================================================
#  CONFIG
# ============================================================
MMD_FILE       = "/Users/sanjeevthakur/Downloads/f818d754-8539-4be6-a099-dfb3a07207d4.mmd"
OUTPUT_DIR     = "/Users/sanjeevthakur/Desktop/pipeline_test/output"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
LLM_MODEL      = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIES        = 3
MAX_QUESTIONS  = 15  # Process up to N questions

# Dedup config
DEDUP_THRESHOLD = 0.92  # cosine similarity above this → duplicate

# Metadata config
CLASS_LEVEL    = 9
SUBJECT        = "Physics"
CHAPTER        = "Sound"
MODE           = "competitive"  # "ncert" or "competitive"

# ============================================================
#  STAGE 1: DELIMITER EXTRACTOR
# ============================================================

HEADING_RE = re.compile(r"^\s*\\(sub)?section\*\{(?P<title>.*)\}\s*$")

def _strip_latex(text: str) -> str:
    t = text.replace(r"\\", " ")
    t = re.sub(r"\\(text|mathbf|mathrm|mathit|bf|it)\s*\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\$(.*?)\$", r"\1", t)
    t = re.sub(r"!\[\]\([^)]*\)", " ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[{}]", " ", t)
    return t

def normalize_heading(title: str) -> str:
    t = _strip_latex(title)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\b([A-Za-z])\s(?=[A-Za-z])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip().lower()

def _kw(*words):
    return lambda h: any(w in h for w in words)

def _rx(*pats):
    compiled = [re.compile(p) for p in pats]
    return lambda h: any(p.search(h) for p in compiled)

CATEGORY_RULES = [
    ("ANSWER_KEY", "answer_key",
     _rx(r"brief explanation", r"^solutions\b", r"^answers?\b", r"^answer key\b",
         r"answers?\s*/\s*hints?", r"hints?\s*(and|&)\s*solution", r"^hints?\b")),
    ("MCQ", "question", _kw("multiple choice", "mcq", "objective type")),
    ("ASSERTION_REASON", "question", _kw("assertion")),
    ("FILL_BLANKS", "question", _kw("fill in the blank")),
    ("TRUE_FALSE", "question", _kw("true / false", "true/false", "true false", "true or false")),
    ("MATCH", "question", _kw("match the following", "match the column", "matching")),
    ("PASSAGE", "question", _kw("passage based", "comprehension")),
    ("INTEGER_NUMERIC", "question", _rx(r"integer\s*/", r"\bnumeric\b", r"\binteger\b.*question")),
    ("VSA", "question", _kw("very short answer")),
    ("SHORT_ANSWER", "question", _kw("short answer")),
    ("LONG_ANSWER", "question", _kw("long answer")),
    ("HOTS", "question", _rx(r"\bhots\b")),
    ("TEXTBOOK_QUESTIONS", "question", _rx(r"text\s*-?\s*book question")),
    ("TEXTBOOK_EXERCISE", "question", _rx(r"text\s*-?\s*book exercise")),
    ("EXEMPLAR", "question", _kw("exemplar")),
    ("SINGLE_OPTION", "question", _kw("single option correct", "single correct")),
    ("MORE_THAN_ONE", "question", _kw("more than one option")),
    ("EXERCISE", "question", _rx(r"^exercises?\b", r"\bexercises?\b")),
    ("QUESTIONS", "question", lambda h: h in ("questions", "question")),
    ("WORKED_EXAMPLE", "exit", _rx(r"\billustrat", r"\bcase study\b", r"^examples?\b",
                                    r"\bsolved example", r"\bworked example")),
    ("WORKED_SOLUTION", "exit", lambda h: h == "solution" or h.startswith("solution")),
    ("ACTIVITY", "exit", _kw("activity")),
    ("SUMMARY", "exit", _kw("summary", "what you have learnt", "what have we discussed")),
    ("THEORY", "exit", _rx(r"^\d+(\.\d+)*\s")),
]

def classify_heading(norm: str):
    for cat, role, matcher in CATEGORY_RULES:
        try:
            if matcher(norm):
                return cat, role
        except re.error:
            continue
    return "OTHER", "neutral"

def find_headings(lines):
    out = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        title = m.group("title")
        norm = normalize_heading(title)
        cat, role = classify_heading(norm)
        out.append({"idx": i, "raw": line.rstrip(), "norm": norm, "cat": cat, "role": role})
    in_answers = False
    for h in out:
        if h["role"] == "answer_key":
            in_answers = True
            continue
        if in_answers and h["role"] == "question":
            h["role"] = "answer_key"
    return out

def extract_question_sections(mmd_path: Path):
    lines = mmd_path.read_text(encoding="utf-8").splitlines()
    headings = find_headings(lines)
    regions = []
    n = len(headings)
    i = 0
    while i < n:
        h = headings[i]
        if h["role"] == "question":
            end = len(lines)
            j = i + 1
            while j < n:
                if headings[j]["role"] in ("question", "answer_key", "exit"):
                    end = headings[j]["idx"]
                    break
                j += 1
            body = "\n".join(lines[h["idx"]+1:end]).strip()
            if body:
                regions.append((h["norm"], h["cat"], body, h["idx"]))
            i = j
        else:
            i += 1
    return regions


# ============================================================
#  STAGE 2: LLM PIPELINE
# ============================================================

def chat_json(client: OpenAI, system: str, user: str, temperature: float = 0.0) -> Optional[dict]:
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            log.warning(f"  LLM attempt {attempt+1}: {str(e)[:100]}")
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return None

STAGE1_SYSTEM = r"""
You are an expert NCERT question extraction specialist (Class 8-10, India).
Extract every individual question from the section block as structured JSON.

RULES:
1. Sub-questions (a)(b)(c) with options → UNROLL into separate questions (Q1a, Q1b, etc.)
2. Roman numeral options (i)(ii)(iii)(iv) → convert to A/B/C/D
3. Options stay attached to their parent question
4. Preserve original stem text exactly
5. Capture inline answers in existing_answer if present

OUTPUT (valid JSON only):
{
  "questions": [
    {"question_id": "Q1", "stem": "...", "options": [{"label": "A", "text": "..."}], "existing_answer": null}
  ]
}
"""

STAGE2_SYSTEM = r"""
You are an expert NCERT question classifier (Class 8-10, India).
Route each question:
- ALREADY_SCQ: Has A/B/C/D options, single correct
- ALREADY_MCQ: Has options, multiple correct
- ALREADY_INTEGER: Expects single numerical answer (calculation result)
- CONVERT_TO_SCQ: Factual/conceptual, one correct answer, can generate options
- BYPASS: Essay/explain/describe/match/fill-blank/true-false/diagram

IMPORTANT: Questions asking "calculate X" with a definite numerical answer → ALREADY_INTEGER
Questions asking "what is X?" with a factual one-word/phrase answer → CONVERT_TO_SCQ

OUTPUT (valid JSON only):
{"routes": [{"question_id": "Q1", "route": "ALREADY_SCQ", "reasoning": "..."}]}
"""

STAGE3_SYSTEM = r"""
You are an expert NCERT question paper setter (Class 8-10, India).
Generate 4 options (A/B/C/D). Place correct answer at RANDOM position (not always A).
BANNED: "None of the above" / "All of the above"
Distractors must be plausible for Class 8-10 students.

OUTPUT (valid JSON only):
{"options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correct_option": "C"}
"""

STAGE4_SYSTEM = r"""
You are an expert NCERT teacher (Class 8-10, India). Solve the question.
For SCQ/MCQ: answer_key = option letter(s). For Integer: answer_key = the number as string.
Write a clear step-by-step solution. No bot phrases.

OUTPUT (valid JSON only):
{"answer_key": "B", "solution_content": "<step-by-step solution>"}
"""

STAGE5_SYSTEM = r"""
You are an NCERT assessment specialist. Rate difficulty:
EASY = direct recall, single fact, definition
MEDIUM = multi-step application or calculation
HARD = synthesis, connecting multiple concepts, HOTS

OUTPUT (valid JSON only):
{"difficulty": "MEDIUM"}
"""


def run_stage2_pipeline(client: OpenAI, sections: list) -> Tuple[list, list]:
    """Run LLM extraction + routing + solving. Returns (processed, bypassed)."""
    all_final = []
    bypass_questions = []

    for sec_idx, (heading, category, body, base_ln) in enumerate(sections):
        if len(all_final) >= MAX_QUESTIONS:
            break

        log.info(f"\n  Section [{sec_idx+1}]: {heading} ({category})")

        # Stage 1: Extract
        data = chat_json(client, STAGE1_SYSTEM,
                         f"Section: {heading}\n\nContent:\n\n{body[:4000]}")
        if not data:
            continue
        questions = data.get("questions", [])
        log.info(f"    Extracted {len(questions)} questions")
        if not questions:
            continue

        # Stage 2: Route
        items = []
        for q in questions:
            opt_str = ""
            if q.get("options"):
                opt_str = " | ".join(f"{o['label']}.{o['text'][:20]}" for o in q["options"][:4])
            items.append(f"question_id: {q['question_id']}\nstem: {q['stem'][:200]}\n{opt_str}")

        route_data = chat_json(client, STAGE2_SYSTEM,
                               "Classify:\n\n" + "\n---\n".join(items))
        route_map = {}
        if route_data:
            for r in route_data.get("routes", []):
                route_map[r["question_id"]] = r.get("route", "BYPASS")

        # Process each
        for q in questions:
            if len(all_final) >= MAX_QUESTIONS:
                break
            qid = q["question_id"]
            stem = q["stem"]
            options = q.get("options", [])
            route = route_map.get(qid, "BYPASS")

            if route == "BYPASS":
                bypass_questions.append({"qid": qid, "stem": stem, "route": route})
                continue

            # Stage 3: Options
            opts_dict = None
            gen_correct = None
            if route == "CONVERT_TO_SCQ":
                opt_data = chat_json(client, STAGE3_SYSTEM, f"Question:\n{stem}", temperature=0.3)
                if opt_data and opt_data.get("options"):
                    opts_dict = opt_data["options"]
                    gen_correct = opt_data.get("correct_option", "")
                else:
                    bypass_questions.append({"qid": qid, "stem": stem, "route": "BYPASS"})
                    continue
            elif options:
                opts_dict = {o["label"]: o["text"] for o in options}

            # Stage 4: Solve
            solve_prompt = f"Question ({route}):\n{stem}\n"
            if opts_dict:
                solve_prompt += "\nOptions:\n" + "\n".join(f"  {k}. {v}" for k, v in sorted(opts_dict.items()))

            solve_data = chat_json(client, STAGE4_SYSTEM, solve_prompt)
            answer_key = gen_correct or ""
            solution = ""
            if solve_data:
                answer_key = str(solve_data.get("answer_key", answer_key)).strip()
                solution = str(solve_data.get("solution_content", "")).strip()

            # Stage 5: Difficulty
            diff_data = chat_json(client, STAGE5_SYSTEM, solve_prompt)
            level = "MEDIUM"
            if diff_data:
                level = str(diff_data.get("difficulty", "MEDIUM")).upper()
                if level not in ("EASY", "MEDIUM", "HARD"):
                    level = "MEDIUM"

            # Build question_content
            qcontent = stem
            if opts_dict:
                qcontent += "\n" + "\n".join(f"({k}) {v}" for k, v in sorted(opts_dict.items()))

            qtype_map = {"ALREADY_SCQ": "scq", "ALREADY_MCQ": "mcq",
                         "ALREADY_INTEGER": "integerQuestion", "CONVERT_TO_SCQ": "scq"}

            all_final.append({
                "qid": f"SOUND_PHY_{qid}",
                "questionType": qtype_map.get(route, "scq"),
                "subject": SUBJECT,
                "question_content": qcontent,
                "answer_key": answer_key,
                "level": level,
                "solution_content": solution,
                "options": opts_dict,
                "route_used": route,
                "section_category": category,
            })
            log.info(f"    ✓ {qid} → {qtype_map.get(route,'scq')} | ans={answer_key} | {level}")

    return all_final, bypass_questions


# ============================================================
#  STAGE 3: CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Clean question/solution text — strip OCR noise, normalize."""
    if not text:
        return ""
    t = text

    # Protect math blocks
    math_blocks = []
    def _protect_math(m):
        math_blocks.append(m.group(0))
        return f"__MATH{len(math_blocks)-1}__"

    t = re.sub(r"\$\$.*?\$\$", _protect_math, t, flags=re.DOTALL)
    t = re.sub(r"\\\[.*?\\\]", _protect_math, t, flags=re.DOTALL)
    t = re.sub(r"\$[^$]+\$", _protect_math, t)

    # Strip leading question numbers like "1.", "Q1.", "(1)"
    t = re.sub(r"^\s*(?:Q?\d+[\.\)]\s*|[\(\[]?\d+[\)\]]\s*)", "", t)

    # Strip MathPix artifacts
    t = re.sub(r"!\[\]\([^)]*\)", "", t)  # empty image refs
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.DOTALL)
    t = re.sub(r"\\captionsetup\{[^}]*\}", "", t)
    t = re.sub(r"\\caption\{[^}]*\}", "", t)
    t = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", "", t)

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()

    # Restore math
    for i, block in enumerate(math_blocks):
        t = t.replace(f"__MATH{i}__", block)

    return t

def clean_questions(questions: list) -> list:
    """Apply cleaning to all questions."""
    cleaned = []
    for q in questions:
        q["question_content"] = clean_text(q["question_content"])
        q["solution_content"] = clean_text(q.get("solution_content", "") or "")
        # Skip if question is empty after cleaning
        if q["question_content"].strip():
            cleaned.append(q)
    return cleaned


# ============================================================
#  STAGE 4: DEDUP (Embedding-based)
# ============================================================

def get_embeddings(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts."""
    BATCH = 20
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vecs = [d.embedding for d in resp.data]
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype="float32")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def dedup_questions(client: OpenAI, questions: list) -> Tuple[list, list]:
    """Remove near-duplicate questions using embedding similarity."""
    if len(questions) <= 1:
        return questions, []

    texts = [q["question_content"] for q in questions]
    log.info(f"    Computing embeddings for {len(texts)} questions...")
    embeddings = get_embeddings(client, texts)

    kept = []
    dropped = []
    kept_indices = []

    for i, q in enumerate(questions):
        is_dup = False
        for j in kept_indices:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > DEDUP_THRESHOLD:
                is_dup = True
                dropped.append({
                    "qid": q["qid"],
                    "duplicate_of": questions[j]["qid"],
                    "similarity": round(sim, 4),
                    "stem_preview": q["question_content"][:80],
                })
                break
        if not is_dup:
            kept.append(q)
            kept_indices.append(i)

    return kept, dropped


# ============================================================
#  STAGE 5: H3 TAGGING + METADATA MAPPING
# ============================================================

def build_question_html(text: str) -> str:
    """Wrap question content in <h3> tags, options as <br/>."""
    lines = text.split("\n")
    html_parts = []
    for line in lines:
        line = line.strip()
        if line:
            html_parts.append(line)
    return "<h3>" + "<br/>".join(html_parts) + "</h3>"

def build_solution_html(text: str) -> str:
    """Wrap solution in <h3> tags."""
    if not text:
        return "<h3></h3>"
    lines = text.strip().split("\n")
    return "<h3>" + "<br/>".join(l.strip() for l in lines if l.strip()) + "</h3>"

def map_to_metadata_v2(questions: list) -> list:
    """Map cleaned questions to Metadata V2 schema (Acadza DB format)."""
    ALL_TYPE_KEYS = ("scq", "mcq", "integerQuestion", "matchQuestion", "subjective", "passageQuestion")

    results = []
    for q in questions:
        qtype = q["questionType"]
        question_html = build_question_html(q["question_content"])
        solution_html = build_solution_html(q.get("solution_content", ""))
        answer = q.get("answer_key", "")

        # Active type container
        container = {
            "question": question_html,
            "solution": solution_html,
            "answer": answer,
            "quesImages": [],
            "solutionImages": [],
        }

        # Empty slots for other types
        empty_slots = {t: {} for t in ALL_TYPE_KEYS if t != qtype}

        # Mode flags
        flags = {}
        if MODE == "competitive":
            flags = {"isNEET": True, "isMHCET": True, "isBoard": False}
        else:
            flags = {"isNEET": False, "isMHCET": False, "isBoard": True}

        record = {
            "qid": q["qid"],
            "questionType": qtype,
            "subject": SUBJECT,
            "chapter": CHAPTER,
            "concept": "TBD",
            "subConcept": "TBD",
            qtype: container,
            **empty_slots,
            "class": CLASS_LEVEL,
            "level": q.get("level", "MEDIUM"),
            "addedByAi": True,
            "taggedBy": "AITAI",
            "tagSubConcept": [],
            "error": 2,
            "isSolvedExample": False,
            "isBuffer": True,
            "isPrevious": False,
            "previousExam": "",
            "previousExamYear": 0,
            "source": f"{MODE}_c{CLASS_LEVEL}_{SUBJECT}_{CHAPTER}",
            **flags,
        }
        results.append(record)

    return results


# ============================================================
#  STAGE 6: MATHML CONVERSION
# ============================================================

def latex_to_mathml(latex_str: str) -> str:
    """Convert a LaTeX expression to MathML."""
    try:
        import latex2mathml.converter
        return latex2mathml.converter.convert(latex_str)
    except Exception:
        return latex_str  # fallback: keep original

def convert_latex_in_text(text: str) -> str:
    """Find all LaTeX expressions in text and convert to MathML."""
    if not text:
        return text

    # Convert display math: $$...$$ and \[...\]
    def _convert_display(m):
        inner = m.group(1) if m.group(1) else m.group(2)
        return latex_to_mathml(inner.strip())

    text = re.sub(r"\$\$(.*?)\$\$", _convert_display, text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: latex_to_mathml(m.group(1).strip()), text, flags=re.DOTALL)

    # Convert inline math: $...$
    def _convert_inline(m):
        inner = m.group(1)
        if not inner.strip():
            return m.group(0)
        return latex_to_mathml(inner.strip())

    text = re.sub(r"\$([^$]+)\$", _convert_inline, text)

    return text

def apply_mathml_conversion(records: list) -> list:
    """Apply MathML conversion to question and solution fields."""
    for rec in records:
        qtype = rec["questionType"]
        container = rec.get(qtype, {})
        if container.get("question"):
            container["question"] = convert_latex_in_text(container["question"])
        if container.get("solution"):
            container["solution"] = convert_latex_in_text(container["solution"])
        rec[qtype] = container
    return records


# ============================================================
#  STAGE 7: CONCEPT TAGGING (GPT-based, no FAISS needed)
# ============================================================

TAGGING_SYSTEM = r"""
You are an NCERT curriculum specialist for Class 9 Physics (India).
Given a question from the chapter "Sound", identify:
1. concept: The main concept being tested (e.g., "Speed of Sound", "Echo", "Frequency and Wavelength")
2. subConcept: A more specific sub-topic (e.g., "Wavelength calculation", "Minimum distance for echo")
3. tagSubConcept: Array of 1-3 relevant sub-concept tags

Use standard NCERT terminology. Be specific.

OUTPUT (valid JSON only):
{
  "concept": "Speed of Sound",
  "subConcept": "Wavelength calculation",
  "tagSubConcept": ["Wavelength", "Frequency", "Wave speed formula"]
}
"""

def tag_concepts(client: OpenAI, records: list) -> list:
    """Use GPT to tag each question with concept/subConcept."""
    log.info(f"    Tagging {len(records)} questions with concepts...")

    for i, rec in enumerate(records):
        qtype = rec["questionType"]
        container = rec.get(qtype, {})
        q_text = container.get("question", "")
        # Strip HTML for tagging
        plain = re.sub(r"<[^>]+>", " ", q_text)
        plain = re.sub(r"\s+", " ", plain).strip()[:300]

        data = chat_json(client, TAGGING_SYSTEM,
                         f"Chapter: Sound\nQuestion: {plain}")
        if data:
            rec["concept"] = data.get("concept", "TBD")
            rec["subConcept"] = data.get("subConcept", "TBD")
            rec["tagSubConcept"] = data.get("tagSubConcept", [])
            log.info(f"      Q{i+1}: {rec['concept']} → {rec['subConcept']}")
        else:
            log.warning(f"      Q{i+1}: tagging failed, keeping TBD")

    return records


# ============================================================
#  MAIN PIPELINE
# ============================================================

def run_full_pipeline():
    t_start = time.time()
    client = OpenAI(api_key=OPENAI_API_KEY)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = Path(MMD_FILE)

    print("\n" + "=" * 70)
    print("  FULL PIPELINE — NCERT Question Processing")
    print("  Input: " + mmd_path.name)
    print("=" * 70)

    # ── STAGE 1 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 1: Delimiter Extraction")
    log.info("━" * 60)

    sections = extract_question_sections(mmd_path)
    log.info(f"  Found {len(sections)} question sections")

    # Pick priority sections (SCQ first, then others)
    priority = []
    for h, cat, body, ln in sections:
        if any(k in h for k in ["single option", "exemplar", "hots",
                                 "integer", "more than one", "text-book"]):
            priority.append((h, cat, body, ln))
    if not priority:
        priority = [(h, cat, body, ln) for h, cat, body, ln in sections[:5]]

    log.info(f"  Priority sections: {len(priority)}")

    # ── STAGE 2 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 2: LLM Question Pipeline (Extract → Route → Solve)")
    log.info("━" * 60)

    processed, bypassed = run_stage2_pipeline(client, priority)
    log.info(f"\n  Result: {len(processed)} processed, {len(bypassed)} bypassed")

    if not processed:
        log.error("No questions processed! Check API key or .mmd content.")
        return

    # ── STAGE 3 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 3: Cleaning")
    log.info("━" * 60)

    cleaned = clean_questions(processed)
    log.info(f"  Cleaned: {len(cleaned)} questions (dropped {len(processed)-len(cleaned)} empty)")

    # ── STAGE 4 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 4: Dedup (Embedding Similarity)")
    log.info("━" * 60)

    deduped, dropped = dedup_questions(client, cleaned)
    log.info(f"  Kept: {len(deduped)}, Dropped as duplicates: {len(dropped)}")
    if dropped:
        for d in dropped:
            log.info(f"    DROPPED: {d['qid']} (sim={d['similarity']} of {d['duplicate_of']})")

    # ── STAGE 5 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 5: H3 Tagging + Metadata V2 Mapping")
    log.info("━" * 60)

    metadata_records = map_to_metadata_v2(deduped)
    log.info(f"  Mapped {len(metadata_records)} records to Metadata V2 schema")

    # ── STAGE 6 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 6: LaTeX → MathML Conversion")
    log.info("━" * 60)

    metadata_records = apply_mathml_conversion(metadata_records)
    log.info(f"  MathML conversion applied to {len(metadata_records)} records")

    # ── STAGE 7 ──────────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("STAGE 7: Concept Tagging (GPT)")
    log.info("━" * 60)

    metadata_records = tag_concepts(client, metadata_records)

    # ── WRITE OUTPUT ─────────────────────────────────────────
    log.info("\n" + "━" * 60)
    log.info("WRITING OUTPUT")
    log.info("━" * 60)

    # Final output
    final_path = out_dir / "sound_final_output.json"
    final_path.write_text(json.dumps(metadata_records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  → {final_path.name} ({len(metadata_records)} questions)")

    # Bypass
    bypass_path = out_dir / "sound_bypass.json"
    bypass_path.write_text(json.dumps(bypassed, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  → {bypass_path.name} ({len(bypassed)} bypass)")

    # Dropped duplicates
    if dropped:
        dup_path = out_dir / "sound_dropped_duplicates.json"
        dup_path.write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"  → {dup_path.name} ({len(dropped)} duplicates)")

    # ── SUMMARY ──────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Time elapsed:    {elapsed}s")
    print(f"  Questions in:    {len(processed)} extracted from .mmd")
    print(f"  After cleaning:  {len(cleaned)}")
    print(f"  After dedup:     {len(deduped)}")
    print(f"  Final output:    {len(metadata_records)} questions")
    print(f"  Bypassed:        {len(bypassed)} (subjective/essay)")
    print(f"  Duplicates:      {len(dropped)}")
    print(f"  Types: SCQ={sum(1 for r in metadata_records if r['questionType']=='scq')}, "
          f"MCQ={sum(1 for r in metadata_records if r['questionType']=='mcq')}, "
          f"Integer={sum(1 for r in metadata_records if r['questionType']=='integerQuestion')}")
    print(f"\n  Output: {out_dir}/")
    print("=" * 70)

    # Print sample
    print("\n\nSAMPLE FINAL OUTPUT (first 2 questions):")
    print("=" * 70)
    print(json.dumps(metadata_records[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_full_pipeline()
