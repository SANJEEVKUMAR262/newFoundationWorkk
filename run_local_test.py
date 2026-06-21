#!/usr/bin/env python3
"""
Local test of the NCERT question pipeline on the Sound chapter .mmd file.
Runs: delimiter_extractor → ncert_answer_pipeline (limited to 10 questions)
→ outputs structured JSON.

Uses SYNCHRONOUS OpenAI client to avoid async issues outside Colab.
"""

from __future__ import annotations
import json, logging, re, time
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ============================================================
#  CONFIG
# ============================================================
MMD_FILE       = "/Users/sanskargupta/Downloads/f818d754-8539-4be6-a099-dfb3a07207d4.mmd"
OUTPUT_DIR     = "/Users/sanskargupta/Desktop/pipeline_test/output"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
LLM_MODEL      = "gpt-4o-mini"
RETRIES        = 3
MAX_QUESTIONS  = 10  # Limit to save API costs

# ============================================================
#  STEP 1: DELIMITER EXTRACTOR (no API needed)
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
                regions.append((h["norm"], body, h["idx"]))
            i = j
        else:
            i += 1
    return regions

# ============================================================
#  STEP 2: LLM PIPELINE (synchronous)
# ============================================================

class FinalQuestion(BaseModel):
    qid: str
    questionType: Literal["scq", "mcq", "integerQuestion"]
    subject: str = "Physics"
    question_content: str
    answer_key: str
    level: Literal["EASY", "MEDIUM", "HARD"]
    solution_content: Optional[str] = None
    options: Optional[Dict[str, str]] = None
    route_used: str = ""

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
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            log.warning(f"  LLM attempt {attempt+1} failed: {str(e)[:120]}")
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return None

# Prompts
STAGE1_SYSTEM = r"""
You are an expert NCERT question extraction specialist (Class 8-10, India).
You receive a raw section block from a textbook exercise and must extract every
individual question inside it as a structured JSON list.

CRITICAL RULES:
1. If a block has sub-questions (a), (b), (c)... each with options → UNROLL them.
2. Convert roman numeral options (i)(ii)(iii)(iv) → A/B/C/D.
3. Options stay attached to their parent question.
4. Preserve the original stem text exactly.
5. If an inline answer is present capture it in existing_answer.

OUTPUT — respond ONLY with valid JSON:
{
  "questions": [
    {
      "question_id": "Q1",
      "stem": "<exact question text>",
      "options": [{"label": "A", "text": "..."}, ...],
      "existing_answer": null
    }
  ]
}
options is an empty list [] when no options exist.
"""

STAGE2_SYSTEM = r"""
You are an expert NCERT question classifier (Class 8-10, India).
For each question decide which ROUTE:
- ALREADY_SCQ: Has A/B/C/D options, single correct answer
- ALREADY_MCQ: Has options, multiple correct answers possible
- ALREADY_INTEGER: Expects a single numerical answer
- CONVERT_TO_SCQ: Subjective but tests one factual concept → can become SCQ
- BYPASS: Essay/open/explain/describe/match/fill-in-blank/true-false

OUTPUT — respond ONLY with valid JSON:
{"routes": [{"question_id": "Q1", "route": "ALREADY_SCQ", "reasoning": "..."}]}
"""

STAGE3_SYSTEM = r"""
You are an expert NCERT question paper setter (Class 8-10, India).
Generate 4 options (A/B/C/D) for this question. Place correct answer at random position.
BANNED: "None of the above" / "All of the above"

OUTPUT — respond ONLY with valid JSON:
{
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_option": "B"
}
"""

STAGE4_SYSTEM = r"""
You are an expert NCERT teacher (Class 8-10, India). Solve the question.
For SCQ/MCQ: answer_key is the option letter(s). For Integer: the number as string.
Write solution directly, no bot phrases.

OUTPUT — respond ONLY with valid JSON:
{"answer_key": "B", "solution_content": "<step-by-step solution>"}
"""

STAGE5_SYSTEM = r"""
You are an NCERT assessment specialist. Rate difficulty: EASY/MEDIUM/HARD.
EASY = direct recall, single fact. MEDIUM = multi-step application. HARD = synthesis/HOTS.

OUTPUT — respond ONLY with valid JSON:
{"difficulty": "EASY"}
"""


def run_pipeline():
    client = OpenAI(api_key=OPENAI_API_KEY)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = Path(MMD_FILE)

    # ── STEP 1: Extract ──────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Extracting question sections from .mmd")
    log.info("=" * 60)

    sections = extract_question_sections(mmd_path)
    log.info(f"Found {len(sections)} question sections")
    for heading, body, ln in sections[:8]:
        log.info(f"  [{ln:4d}] {heading[:60]}")

    # Pick priority sections (SCQ, textbook, exemplar, integer, etc.)
    priority_sections = []
    for heading, body, ln in sections:
        if any(k in heading for k in ["single option", "text-book question", "text-book exercise",
                                       "exemplar", "hots", "integer", "more than one"]):
            priority_sections.append((heading, body, ln))

    if not priority_sections:
        priority_sections = sections[:3]

    log.info(f"\nPriority sections: {len(priority_sections)} (processing up to {MAX_QUESTIONS} questions)")

    # ── STEP 2: LLM Pipeline ─────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2: LLM Extraction → Routing → Solving")
    log.info("=" * 60)

    all_final: List[FinalQuestion] = []
    bypass_questions = []

    for sec_idx, (heading, body, base_ln) in enumerate(priority_sections):
        if len(all_final) >= MAX_QUESTIONS:
            break

        log.info(f"\n{'─'*50}")
        log.info(f"Section [{sec_idx+1}]: {heading}")
        log.info(f"{'─'*50}")

        # Stage 1: Extract questions
        user_msg = f"Section heading: {heading}\n\nRaw content:\n\n{body[:4000]}"
        data = chat_json(client, STAGE1_SYSTEM, user_msg)
        if not data:
            log.warning("  Stage 1 failed — skipping section")
            continue

        questions = data.get("questions", [])
        log.info(f"  Stage 1: Extracted {len(questions)} questions")
        if not questions:
            continue

        # Stage 2: Route all questions in this section
        items = []
        for q in questions:
            opt_preview = ""
            if q.get("options"):
                opt_preview = "  Options: " + " | ".join(
                    f"{o['label']}. {o['text'][:25]}" for o in q["options"][:4])
            items.append(f"question_id: {q['question_id']}\nstem: {q['stem'][:250]}\n{opt_preview}")

        route_msg = "Classify each question:\n\n" + "\n\n---\n\n".join(items)
        route_data = chat_json(client, STAGE2_SYSTEM, route_msg)

        route_map = {}
        if route_data:
            for r in route_data.get("routes", []):
                route_map[r["question_id"]] = r.get("route", "BYPASS")
            log.info(f"  Stage 2: Routes assigned")
            for qid, route in route_map.items():
                log.info(f"    {qid:8s} → {route}")

        # Process each question (Stages 3-5)
        for q in questions:
            if len(all_final) >= MAX_QUESTIONS:
                break

            qid = q["question_id"]
            stem = q["stem"]
            options = q.get("options", [])
            route = route_map.get(qid, "BYPASS")

            if route == "BYPASS":
                bypass_questions.append({"qid": qid, "stem": stem[:100], "route": route})
                continue

            log.info(f"\n  Processing {qid} ({route})...")

            # Stage 3: Generate options if CONVERT_TO_SCQ
            opts_dict = None
            gen_correct = None
            if route == "CONVERT_TO_SCQ":
                opt_data = chat_json(client, STAGE3_SYSTEM,
                                     f"Question:\n{stem}", temperature=0.3)
                if opt_data and opt_data.get("options"):
                    opts_dict = opt_data["options"]
                    gen_correct = opt_data.get("correct_option", "")
                    log.info(f"    Stage 3: Options generated, correct={gen_correct}")
                else:
                    bypass_questions.append({"qid": qid, "stem": stem[:100], "route": "BYPASS (gen failed)"})
                    continue
            elif options:
                opts_dict = {o["label"]: o["text"] for o in options}

            # Stage 4: Solve
            solve_prompt = f"Question ({route}):\n{stem}\n"
            if opts_dict:
                solve_prompt += "\nOptions:\n" + "\n".join(
                    f"  {k}. {v}" for k, v in sorted(opts_dict.items()))

            solve_data = chat_json(client, STAGE4_SYSTEM, solve_prompt)
            answer_key = ""
            solution = ""
            if solve_data:
                answer_key = str(solve_data.get("answer_key", gen_correct or "")).strip()
                solution = str(solve_data.get("solution_content", "")).strip()
            elif gen_correct:
                answer_key = gen_correct

            # Stage 5: Difficulty
            diff_data = chat_json(client, STAGE5_SYSTEM, solve_prompt)
            level = "MEDIUM"
            if diff_data:
                level = str(diff_data.get("difficulty", "MEDIUM")).upper()
                if level not in ("EASY", "MEDIUM", "HARD"):
                    level = "MEDIUM"

            # Build question_content with embedded options
            qcontent = stem
            if opts_dict:
                qcontent += "\n" + "\n".join(f"({k}) {v}" for k, v in sorted(opts_dict.items()))

            # Determine type
            qtype_map = {
                "ALREADY_SCQ": "scq",
                "ALREADY_MCQ": "mcq",
                "ALREADY_INTEGER": "integerQuestion",
                "CONVERT_TO_SCQ": "scq",
            }
            qtype = qtype_map.get(route, "scq")

            final = FinalQuestion(
                qid=f"SOUND_PHY_{qid}",
                questionType=qtype,
                question_content=qcontent,
                answer_key=answer_key,
                level=level,
                solution_content=solution,
                options=opts_dict,
                route_used=route,
            )
            all_final.append(final)
            log.info(f"    ✓ {qid} → {qtype} | ans={answer_key} | level={level}")

    # ── STEP 3: Write Output ─────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3: Writing Output")
    log.info("=" * 60)

    output_json = [q.model_dump() for q in all_final]
    json_path = out_dir / "sound_structured.json"
    json_path.write_text(json.dumps(output_json, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  → {json_path.name} ({len(all_final)} questions)")

    bypass_path = out_dir / "sound_bypass.json"
    bypass_path.write_text(json.dumps(bypass_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  → {bypass_path.name} ({len(bypass_questions)} bypass)")

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("SUMMARY")
    log.info(f"  Total processed: {len(all_final)} questions")
    log.info(f"  Bypassed: {len(bypass_questions)} questions")
    log.info(f"  Types: SCQ={sum(1 for q in all_final if q.questionType=='scq')}, "
             f"MCQ={sum(1 for q in all_final if q.questionType=='mcq')}, "
             f"Integer={sum(1 for q in all_final if q.questionType=='integerQuestion')}")
    log.info(f"  Output: {out_dir}/")
    log.info(f"{'=' * 60}")

    # Print sample
    print("\n\n" + "=" * 60)
    print("SAMPLE OUTPUT JSON (first 3 questions):")
    print("=" * 60)
    print(json.dumps(output_json[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_pipeline()