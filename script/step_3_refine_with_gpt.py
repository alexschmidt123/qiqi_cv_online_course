"""
Step 3 (optional): Use ChatGPT to refine OCR text and produce the final unified gaze table.

For one video (one output folder):
- Input (in output/<video_name>/):
    - full_text.txt
    - gaze_with_section.csv
- Output (in output/<video_name>/):
    - full_text_combined.txt      # original OCR + GPT-optimized text
    - final_gaze_table.csv        # frame_idx, timestamp_sec, has_gaze, gaze_x, gaze_y, section_name
    - gaze_section_gpt_report.md  # short QA report on gaze→section mapping

Usage (from project root):
  python script/step_3_refine_with_gpt.py --output-dir output/R10,P8_2

Requires:
  - OPENAI_API_KEY in environment or in .env at project root.
  - `openai` and `python-dotenv` available (see requirements.txt).
"""
import argparse
import csv
import difflib
import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default section time bucket (seconds). Override with --interval when running.
DEFAULT_INTERVAL_SEC = 1.0


def _load_full_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_gaze_with_section(path: str) -> Tuple[Counter, int]:
    """
    Return (section_counts, total_rows)
    Only counts non-empty section values.
    """
    counts: Counter = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            sec = (row.get("section") or "").strip()
            # Don't treat "none" as a real section.
            if sec and sec.lower() != "none":
                counts[sec] += 1
    return counts, total


def _load_gaze_rows(path: str) -> List[Dict[str, str]]:
    """Load all rows from gaze_with_section.csv for per-bucket section assignment."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _build_section_per_bucket(rows: List[Dict[str, str]], interval_sec: float) -> Dict[int, str]:
    """
    Build one section label per time bucket.

    Step 2 assigns section per frame. Within one second there are many frames (e.g. ~30 at 30fps).
    A few frames may be "none" (gaze jitter, brief off-article) while most frames show the real
    section. We must NOT treat "any single none in the bucket" as forcing the whole bucket to
    none — that was collapsing almost every bucket to none in final_gaze_table.csv.

    Rules:
    - Collect all section labels per bucket.
    - If a strict majority of frames are "none", the bucket is "none".
    - Otherwise use the plurality among non-"none" labels (mode).
    - Forward-fill gaps between min/max bucket using the last non-"none" section; "none" breaks
      the chain (same as before).
    - Backward-fill empty slots before the first non-"none" bucket with that first heading.
    """
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    bucket_to_secs: Dict[int, List[str]] = defaultdict(list)
    for row in rows:
        try:
            ts = float(row.get("timestamp_sec", 0))
        except (TypeError, ValueError):
            continue
        bucket = int(ts / interval_sec)
        sec = (row.get("section") or "").strip()
        bucket_to_secs[bucket].append(sec if sec else "none")

    if not bucket_to_secs:
        return {}

    min_bucket = min(bucket_to_secs.keys())
    max_bucket = max(bucket_to_secs.keys())

    bucket_primary: Dict[int, str] = {}
    for b, secs in bucket_to_secs.items():
        n = len(secs)
        n_none = sum(1 for s in secs if not s or s.lower() == "none")
        non_none = [s for s in secs if s and s.lower() != "none"]
        if not non_none:
            bucket_primary[b] = "none"
        elif n_none > n / 2:
            bucket_primary[b] = "none"
        else:
            bucket_primary[b] = Counter(non_none).most_common(1)[0][0]

    filled: Dict[int, str] = {}
    last_heading = ""
    for b in range(min_bucket, max_bucket + 1):
        if b not in bucket_primary:
            filled[b] = last_heading
            continue
        if bucket_primary[b] == "none":
            filled[b] = "none"
            last_heading = ""
        else:
            filled[b] = bucket_primary[b]
            last_heading = bucket_primary[b]

    first_non_none = next(
        (b for b in range(min_bucket, max_bucket + 1) if filled.get(b) and filled[b] != "none"),
        None,
    )
    if first_non_none is not None:
        heading = filled[first_non_none]
        for b in range(min_bucket, first_non_none):
            if not filled.get(b):
                filled[b] = heading

    return filled


def _get_openai_client() -> OpenAI:
    # Try to load .env in project root
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or to .env in the project root."
        )
    return OpenAI(api_key=api_key)


def _build_prompt(video_name: str, full_text: str, section_counts: Counter, total_rows: int) -> str:
    top_sections = section_counts.most_common()
    counts_str = "\n".join(
        f"- {name}: {count} gaze frames" for name, count in top_sections
    ) or "(no sections with gazes)"
    prompt = f"""
You are reviewing an OCR'd article and a gaze-to-section mapping for an online course video.

VIDEO NAME: {video_name}

PART 1 — ARTICLE ONLY (for section judgement)
Below is raw OCR from the video frames (it includes navigation, URLs, buttons, and the scrollable article). Extract and output ONLY the scrollable article content that the reader sees: the main title, ALL-CAPS section headings, and body paragraphs. Omit entirely: page navigation (e.g. About Me, Research, Teaching), URLs, copyright lines, "REPLAY", operation bar text, and any button or UI labels. Output a single, clean, readable article with clear section headings so it can be used as the reference for gaze→section mapping.

RAW OCR (may contain nav/UI; extract article only):
---
{full_text.strip()}
---

PART 2 — GAZE→SECTION QA
We also have gaze data: each video frame with a visible gaze point was assigned to a section (based on the nearest ALL-CAPS heading above the gaze).
Here are the total gaze frames per section (only non-empty assignments), out of {total_rows} total rows:

{counts_str}

Based on the cleaned article text and these counts:
- Comment on whether the distribution of gazes across sections looks plausible.
- Point out any sections whose gaze counts seem suspiciously low or high given the text length and importance.
- Suggest 1–3 simple rules we could use to further improve the section assignment (without changing model code, just high-level ideas).

RESPONSE FORMAT:
1. Start with a section titled "CLEANED ARTICLE TEXT" and provide only the extracted article (main title, section headings, body). No navigation or UI text.
2. Then a section titled "GAZE→SECTION QA" with 2–4 short bullet points of observations/recommendations.
"""
    return prompt


def _build_refine_sections_prompt(
    cleaned_text: str, section_per_bucket: Dict[int, str], interval_sec: float
) -> str:
    """Build prompt for GPT to refine section name per time bucket."""
    if not section_per_bucket:
        return ""
    buckets = sorted(section_per_bucket.keys())
    lines = [
        f"Bucket {b} ({b * interval_sec:.1f}s–{(b + 1) * interval_sec:.1f}s): {section_per_bucket[b]}"
        for b in buckets
    ]
    return f"""
Using the article text below, refine the section assignment for each time bucket (one section every {interval_sec:.0f} second(s)). For each bucket: use the exact ALL-CAPS heading from the article, or "main title" when the reader is on the main title (keep as is if preliminary is already "main title"), or "none" for gaze outside the scrollable article. Correct OCR typos to match the article headings.

ARTICLE TEXT:
---
{cleaned_text.strip()[:8000]}
---

PRELIMINARY SECTION PER BUCKET:
{chr(10).join(lines)}

Reply with exactly {len(buckets)} lines: line 1 = section for bucket {buckets[0]}, line 2 = section for bucket {buckets[1]}, etc. Use the exact ALL-CAPS heading from the article or "none". No other text.
"""


def _extract_article_body_from_cleaned(cleaned_part: str) -> str:
    """Take the CLEANED ARTICLE TEXT section only; strip header/fence lines for a neat article file."""
    s = cleaned_part.strip()
    # Truncate at GAZE→SECTION QA so we don't include the QA section
    qa_marker = "GAZE→SECTION QA"
    idx = s.find(qa_marker)
    if idx != -1:
        s = s[:idx].strip()
    # Remove common header/fence lines at the start
    drop = {"cleaned article text", "---", "# cleaned article text", ""}
    lines = s.splitlines()
    start = 0
    for i, line in enumerate(lines):
        t = line.strip().lower()
        if t in drop or (t.startswith("#") and "article" in t):
            start = i + 1
        elif t and t not in drop:
            break
    return "\n".join(lines[start:]).strip()


def _extract_headings_from_cleaned_text(cleaned_text: str) -> Set[str]:
    """Extract ALL-CAPS section headings from GPT-cleaned article (e.g. 'STRENGTHS:', 'MISCONCEPTIONS:')."""
    headings: Set[str] = set()
    for line in cleaned_text.splitlines():
        s = line.strip()
        if not s or len(s) < 3:
            continue
        # Allow trailing colon; require mostly letters and caps
        s_nocolon = s.rstrip(":")
        letters = [c for c in s_nocolon if c.isalpha()]
        if len(letters) < 2:
            continue
        if all(c.isupper() for c in letters):
            headings.add(s)  # keep original form including colon if present
    return headings


def _correct_section_name_to_heading(section_name: str, canonical_headings: Set[str]) -> str:
    """If section_name looks like an OCR typo of a canonical heading, return the correct one."""
    s = (section_name or "").strip()
    if not s or not canonical_headings:
        return s
    if s.lower() == "main title":
        return s
    if s.lower() == "none":
        return "none"
    if s in canonical_headings:
        return s
    # Normalize for fuzzy match: strip trailing colon, uppercase
    s_norm = s.rstrip(":").strip().upper()
    candidates = [h.rstrip(":").strip().upper() for h in canonical_headings]
    matches = difflib.get_close_matches(s_norm, candidates, n=1, cutoff=0.6)
    if matches:
        # Map back to original form (with colon if canonical has it)
        for h in canonical_headings:
            if h.rstrip(":").strip().upper() == matches[0]:
                return h
    return s


def _parse_refined_sections(response: str, buckets: List[int]) -> Dict[int, str]:
    """Parse GPT reply into bucket -> section_name. One section per line in order of buckets."""
    refined: Dict[int, str] = {}
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    for i, b in enumerate(buckets):
        if i < len(lines):
            name = lines[i].strip()
            if name.lower() == "none":
                refined[b] = "none"
            else:
                refined[b] = name
        # Missing line: omit key so final_gaze_table falls back to step-2 preliminary (not "none").
    return refined


def run_refinement(
    output_dir: str, model: str = "gpt-4o-mini", interval_sec: float = DEFAULT_INTERVAL_SEC
) -> Dict[str, str]:
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_bucket_csv = os.path.join(debug_dir, "step3_bucket_debug.csv")

    full_text_path = os.path.join(output_dir, "full_text.txt")
    gaze_with_section_path = os.path.join(output_dir, "gaze_with_section.csv")
    if not os.path.isfile(full_text_path):
        raise FileNotFoundError(f"full_text.txt not found in {output_dir}")
    if not os.path.isfile(gaze_with_section_path):
        raise FileNotFoundError(f"gaze_with_section.csv not found in {output_dir}")

    video_name = os.path.basename(os.path.normpath(output_dir))
    full_text = _load_full_text(full_text_path)
    section_counts, total_rows = _load_gaze_with_section(gaze_with_section_path)
    gaze_rows = _load_gaze_rows(gaze_with_section_path)
    section_per_bucket = _build_section_per_bucket(gaze_rows, interval_sec)

    client = _get_openai_client()
    prompt = _build_prompt(video_name, full_text, section_counts, total_rows)

    print(f"Calling OpenAI model {model} for {video_name} (this may take a few seconds)...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )
    content = (resp.choices[0].message.content or "").strip()

    combined_text_path = os.path.join(output_dir, "full_text_combined.txt")
    report_path = os.path.join(output_dir, "gaze_section_gpt_report.md")

    # Heuristic: split cleaned text and QA section
    marker = "GAZE→SECTION QA"
    idx = content.find(marker)
    if idx != -1:
        cleaned_part = content[:idx].strip()
        qa_part = content[idx:].strip()
    else:
        cleaned_part = content
        qa_part = ""

    # Write only the optimized article (reference for section judgement); omit original OCR and nav/UI
    article_body = _extract_article_body_from_cleaned(cleaned_part)
    with open(combined_text_path, "w", encoding="utf-8") as f:
        f.write("===== ARTICLE (reference for section judgement) =====\n")
        f.write("Main title, ALL-CAPS section headings, and body only. No navigation or operation bars.\n\n")
        f.write(article_body.strip())
        if article_body.strip():
            f.write("\n")

    # Write QA report
    with open(report_path, "w", encoding="utf-8") as f:
        if qa_part:
            f.write(qa_part + ("\n" if not qa_part.endswith("\n") else ""))
        else:
            f.write("# GAZE→SECTION QA\n\n(No separate QA section could be parsed from the model response.)\n")

    # Refine section per time bucket via second GPT call
    refined_section_per_bucket: Dict[int, str] = {}
    if section_per_bucket:
        refine_prompt = _build_refine_sections_prompt(
            cleaned_part, section_per_bucket, interval_sec
        )
        if refine_prompt:
            print("Refining section names per time bucket (one GPT call)...")
            ref_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": refine_prompt}],
                max_tokens=2048,
                temperature=0.1,
            )
            ref_content = (ref_resp.choices[0].message.content or "").strip()
            buckets = sorted(section_per_bucket.keys())
            refined_section_per_bucket = _parse_refined_sections(ref_content, buckets)

    # Save per-bucket refinement trace for debugging.
    with open(debug_bucket_csv, "w", newline="", encoding="utf-8") as fdbg:
        wdbg = csv.writer(fdbg)
        wdbg.writerow([
            "bucket",
            "time_start_sec",
            "time_end_sec",
            "section_before_gpt",
            "section_after_gpt_raw",
            "section_final",
        ])
        for b in sorted(section_per_bucket.keys()):
            pre = section_per_bucket.get(b, "")
            post_raw = refined_section_per_bucket.get(b, "")
            final_sec = post_raw or pre
            wdbg.writerow([b, b * interval_sec, (b + 1) * interval_sec, pre, post_raw, final_sec])

    # Correct OCR typos in section names using headings extracted from cleaned article
    canonical_headings = _extract_headings_from_cleaned_text(cleaned_part)

    # Build final_gaze_table.csv: one section per bucket applied to all frames in that bucket
    final_csv_path = os.path.join(output_dir, "final_gaze_table.csv")
    fieldnames = [
        "frame_idx",
        "timestamp_sec",
        "has_gaze",
        "gaze_x",
        "gaze_y",
        "section_name",
    ]
    with open(final_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in gaze_rows:
            frame_idx = row.get("frame_idx", "")
            ts = row.get("timestamp_sec", "")
            x_s = (row.get("x_bl") or "").strip()
            y_s = (row.get("y_bl") or "").strip()
            has_gaze = "1" if x_s and y_s else "0"
            try:
                t = float(ts)
            except (TypeError, ValueError):
                t = 0.0
            bucket = int(t / interval_sec)
            section_name = refined_section_per_bucket.get(bucket) or section_per_bucket.get(bucket, "")
            section_name = _correct_section_name_to_heading(section_name, canonical_headings)
            # Normalize empty/unspecified to explicit "none" so outputs always use one of:
            # - a concrete ALL-CAPS section heading
            # - "main title"
            # - "none"
            if not section_name or not str(section_name).strip():
                section_name = "none"
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": ts,
                    "has_gaze": has_gaze,
                    "gaze_x": x_s,
                    "gaze_y": y_s,
                    "section_name": section_name,
                }
            )

    return {
        "full_text_combined": combined_text_path,
        "gpt_report": report_path,
        "final_gaze_table": final_csv_path,
        "debug_bucket_csv": debug_bucket_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 (optional): refine OCR text and build final_gaze_table.csv using ChatGPT."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Per-video output folder (e.g. output/R10,P8_2). Must contain full_text.txt and gaze_with_section.csv.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        metavar="SEC",
        help="Section time bucket in seconds; one section per bucket (default: 1.0).",
    )
    args = parser.parse_args()

    stats = run_refinement(
        os.path.abspath(args.output_dir),
        model=args.model,
        interval_sec=args.interval,
    )
    print("Wrote combined text to:", stats["full_text_combined"])
    print("Wrote GPT QA report to:", stats["gpt_report"])
    print("Wrote final gaze table to:", stats["final_gaze_table"])
    print("Wrote step3 debug bucket CSV to:", stats["debug_bucket_csv"])


if __name__ == "__main__":
    main()
