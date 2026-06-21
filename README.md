# NCERT Question Pipeline

A full 7-stage pipeline that processes MathPix `.mmd` files (OCR output from NCERT textbooks) into structured, tagged, and database-ready question banks.

## Pipeline Stages

```
.mmd file → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7 → Final JSON
```

| Stage | Name | Description |
|-------|------|-------------|
| 1 | **Delimiter Extraction** | Parse `.mmd`, identify exercise/answer sections using heading taxonomy |
| 2 | **LLM Question Pipeline** | Extract questions → Route (SCQ/MCQ/Integer/Bypass) → Generate options → Solve → Rate difficulty |
| 3 | **Cleaning** | Strip OCR noise, MathPix artifacts, normalize formatting |
| 4 | **Dedup (Embeddings)** | Create embeddings, find near-duplicates via cosine similarity, drop them |
| 5 | **H3 Tagging + Metadata** | Map to Acadza DB schema with `<h3>` wrapping, mode flags, class level |
| 6 | **LaTeX → MathML** | Convert all LaTeX math expressions to MathML |
| 7 | **Concept Tagging** | GPT-based tagging with chapter/concept/subConcept |

## Requirements

```bash
pip install openai pydantic numpy latex2mathml
```

## Usage

1. Set your OpenAI API key in the `OPENAI_API_KEY` variable in the script
2. Set `MMD_FILE` to point to your `.mmd` file
3. Run:

```bash
python run_full_pipeline.py
```

## Configuration

Edit the CONFIG section at the top of `run_full_pipeline.py`:

```python
MMD_FILE        = "/path/to/your/file.mmd"
OUTPUT_DIR      = "/path/to/output"
OPENAI_API_KEY  = "YOUR_OPENAI_API_KEY_HERE"
LLM_MODEL       = "gpt-4o-mini"
MAX_QUESTIONS   = 15          # Limit to save API costs
DEDUP_THRESHOLD = 0.92        # Cosine similarity threshold for dedup
CLASS_LEVEL     = 9
SUBJECT         = "Physics"
CHAPTER         = "Sound"
MODE            = "competitive"  # "ncert" or "competitive"
```

## Output

The pipeline produces:

```
output/
├── sound_final_output.json          ← Final questions (Acadza DB schema)
├── sound_bypass.json                ← Bypassed subjective/essay questions
└── sound_dropped_duplicates.json    ← Dropped duplicate records
```

## Output JSON Schema (per question)

```json
{
  "qid": "SOUND_PHY_Q3",
  "questionType": "scq",
  "subject": "Physics",
  "chapter": "Sound",
  "concept": "Nature of Sound Waves",
  "subConcept": "Mechanical Waves",
  "scq": {
    "question": "<h3>Why are sound waves called mechanical waves?<br/>(A) ...<br/>(B) ...</h3>",
    "solution": "<h3>Sound waves require a medium to propagate...</h3>",
    "answer": "B",
    "quesImages": [],
    "solutionImages": []
  },
  "mcq": {},
  "integerQuestion": {},
  "matchQuestion": {},
  "subjective": {},
  "passageQuestion": {},
  "class": 9,
  "level": "EASY",
  "addedByAi": true,
  "taggedBy": "AITAI",
  "tagSubConcept": ["Medium", "Wave propagation", "Sound characteristics"],
  "error": 2,
  "isNEET": true,
  "isMHCET": true,
  "isBoard": false,
  "source": "competitive_c9_Physics_Sound"
}
```

## Files

- `run_full_pipeline.py` — Complete 7-stage pipeline (recommended)
- `run_local_test.py` — Simplified 3-stage test (extract + route + solve only)
- `output/` — Sample output from processing a Sound chapter (Class 9 Physics)

## Sample Results (Sound Chapter)

- **14 questions** processed (8 SCQ + 6 Integer)
- **14 questions** bypassed (subjective/essay)
- **1 duplicate** detected and dropped (94.3% similarity)
- All questions tagged with concepts like "Speed of Sound", "Echo", "Frequency and Wavelength"