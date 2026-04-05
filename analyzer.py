"""Core analysis pipeline: frames -> VLM -> rubric scoring."""

import json
import logging
from pathlib import Path

import vlm_client
import rubric_manager
import config

logger = logging.getLogger(__name__)


def identify_characters(
    frame_paths: list[Path],
    model: str | None = None,
    progress_callback=None,
) -> dict:
    """
    First pass: identify and track team members across frames.
    Uses two rounds — initial identification then verification — for consistency.
    Returns a character registry with robust visual descriptors.
    """
    # ── Round 1: initial ID from spread of frames ─────────────────────
    n_id_frames = config.CHARACTER_ID_FRAMES
    sample_indices = _spread_sample(len(frame_paths), min(n_id_frames, len(frame_paths)))
    sample_frames = [frame_paths[i] for i in sample_indices]

    prompt_round1 = """Analyze these frames from an emergency room simulation video.
Identify each distinct person visible across the frames.

IMPORTANT: People must be distinguished by PERSISTENT visual features — not by
what they are doing in a single frame, since actions change over time.

For EACH person, provide ALL of the following anchoring features:
1. **Label**: A role-based label (e.g., "Team Leader", "Nurse 1", "Airway Manager")
2. **Body position**: Where they consistently stand relative to the patient/bed
   (e.g., "head of bed", "right side of bed", "foot of bed")
3. **Clothing**: Scrub color, gown, glove color, any visible ID badge or lanyard color
4. **Hair/build**: Hair color/style, approximate build, height relative to others
5. **Equipment**: Any equipment they consistently handle (stethoscope, clipboard, airway cart)
6. **Apparent role**: What their actions suggest about their team role

Return a JSON object:
{
  "characters": [
    {
      "label": "Team Leader",
      "visual_id": "Tall person in dark blue scrubs with short brown hair, standing at foot of bed, wearing red lanyard",
      "body_position": "foot of bed, facing team",
      "clothing": "dark blue scrubs, red lanyard, blue gloves",
      "distinguishing_features": "short brown hair, glasses, taller than others",
      "apparent_role": "Leading resuscitation, calling out orders, not performing hands-on tasks"
    }
  ],
  "scene_description": "Brief description of the simulation setup and room layout"
}

Return ONLY valid JSON."""

    if progress_callback:
        progress_callback("Identifying team members (round 1 of 2)...")

    try:
        raw = vlm_client.ask_vision(prompt_round1, sample_frames, model=model, max_tokens=2500)
        logger.info("Character ID round 1 raw response (%d chars): %s", len(raw), raw[:500])
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        round1 = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Character ID round 1 — VLM returned non-JSON: %s\nRaw: %s", e, raw[:500])
        return {"characters": [], "scene_description": "VLM response was not valid JSON."}
    except Exception as e:
        logger.warning("Character identification round 1 failed: %s", e)
        return {"characters": [], "scene_description": "Could not identify characters."}

    # ── Round 2: verify with different frames ─────────────────────────
    # Use a different sample to cross-check and stabilize labels
    verify_indices = _spread_sample(len(frame_paths), min(n_id_frames, len(frame_paths)))
    # Offset to pick different frames than round 1
    offset = max(1, len(frame_paths) // (2 * max(len(verify_indices), 1)))
    verify_indices = [min(i + offset, len(frame_paths) - 1) for i in verify_indices]
    verify_frames = [frame_paths[i] for i in verify_indices]

    char_summary = json.dumps(round1.get("characters", []), indent=2)
    prompt_round2 = f"""I previously identified these team members in an ER simulation video:

{char_summary}

Now look at these NEW frames from the same video. For each person visible:
1. Match them to the character list above using their CLOTHING, BODY POSITION, and PHYSICAL FEATURES — not their current action
2. If someone doesn't match any existing character, add them as a new entry
3. If any character descriptions need correction, update them

Return the FINAL corrected character list as JSON:
{{
  "characters": [
    {{
      "label": "...",
      "visual_id": "single-sentence summary combining clothing + position + build",
      "body_position": "...",
      "clothing": "...",
      "distinguishing_features": "...",
      "apparent_role": "..."
    }}
  ],
  "scene_description": "..."
}}

Return ONLY valid JSON."""

    if progress_callback:
        progress_callback("Verifying team members (round 2 of 2)...")

    try:
        raw = vlm_client.ask_vision(prompt_round2, verify_frames, model=model, max_tokens=2500)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw)
        # Carry over scene_description from round 1 if missing
        if not result.get("scene_description"):
            result["scene_description"] = round1.get("scene_description", "")
        return result
    except Exception as e:
        logger.warning("Character verification round 2 failed: %s, using round 1 results.", e)
        return round1


def analyze_frames_in_batches(
    frame_paths: list[Path],
    character_info: dict,
    model: str | None = None,
    progress_callback=None,
) -> list[dict]:
    """
    Second pass: analyze frames in batches, describing clinical actions
    with character tracking context.
    """
    batch_size = config.MAX_FRAMES_PER_BATCH
    batches = [
        frame_paths[i : i + batch_size]
        for i in range(0, len(frame_paths), batch_size)
    ]

    char_context = ""
    if character_info.get("characters"):
        char_lines = []
        for c in character_info["characters"]:
            parts = [c.get("label", "Unknown")]
            if c.get("clothing"):
                parts.append(f"Clothing: {c['clothing']}")
            if c.get("body_position"):
                parts.append(f"Position: {c['body_position']}")
            if c.get("distinguishing_features"):
                parts.append(f"Features: {c['distinguishing_features']}")
            # Fall back to visual_id if detailed fields aren't present
            if len(parts) == 1 and c.get("visual_id"):
                parts.append(c["visual_id"])
            char_lines.append("- " + " | ".join(parts))
        char_context = (
            "KNOWN TEAM MEMBERS — identify by CLOTHING and BODY POSITION, not by action:\n"
            + "\n".join(char_lines)
            + "\n\nIMPORTANT: Use the labels above consistently. Match people by their "
            "clothing/position/build, NOT by what they happen to be doing in a given frame."
        )

    all_analyses = []
    for batch_idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(
                f"Analyzing frames batch {batch_idx + 1}/{len(batches)}..."
            )

        # Extract timestamps from filenames
        timestamps = []
        for p in batch:
            parts = p.stem.split("_t")
            ts = parts[-1].rstrip("s") if len(parts) > 1 else "?"
            timestamps.append(ts)

        prompt = f"""You are an emergency medicine simulation evaluator reviewing video frames
from a mock resuscitation drill.

{char_context}

These frames are from timestamps: {', '.join(t + 's' for t in timestamps)}

For each frame, describe:
1. What clinical actions are being performed
2. Which team member(s) are involved (use their labels from the character list)
3. Any communication or teamwork behaviors visible
4. Any concerns about technique, safety, or protocol adherence

Return a JSON array with one entry per frame:
[
  {{
    "timestamp": "...",
    "actions": ["action 1", "action 2"],
    "characters_involved": ["Team Leader", "Nurse 1"],
    "communication_observed": "description or null",
    "concerns": "description or null"
  }}
]

Return ONLY valid JSON."""

        try:
            raw = vlm_client.ask_vision(prompt, batch, model=model, max_tokens=2000)
            logger.info("Batch %d raw response (%d chars): %s", batch_idx, len(raw), raw[:300])
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            batch_result = json.loads(raw)
            if isinstance(batch_result, list):
                all_analyses.extend(batch_result)
            else:
                all_analyses.append(batch_result)
        except json.JSONDecodeError as e:
            logger.warning("Batch %d — VLM returned non-JSON: %s\nRaw: %s", batch_idx, e, raw[:300])
        except Exception as e:
            logger.warning("Batch %d analysis failed: %s", batch_idx, e)
            for ts in timestamps:
                all_analyses.append({
                    "timestamp": ts,
                    "actions": ["[Analysis failed for this frame]"],
                    "characters_involved": [],
                    "communication_observed": None,
                    "concerns": None,
                })

    return all_analyses


def score_rubric(
    rubric: dict,
    frame_analyses: list[dict],
    character_info: dict,
    transcript: str,
    model: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Third pass: score each rubric item based on accumulated evidence.
    Scores one category at a time to stay within context limits.
    """
    # Build compact evidence summary
    evidence_lines = []
    for fa in frame_analyses:
        ts = fa.get("timestamp", "?")
        actions = ", ".join(fa.get("actions", []))
        comm = fa.get("communication_observed") or ""
        line = f"[{ts}s] {actions}"
        if comm:
            line += f" | Comm: {comm}"
        evidence_lines.append(line)

    evidence_summary = "\n".join(evidence_lines[:30])
    transcript_excerpt = transcript[:1000] if transcript else "(no transcript)"

    # Score one category at a time to keep prompts short
    all_scores: dict[str, dict] = {}
    categories = rubric.get("categories", [])

    for cat_idx, cat in enumerate(categories):
        if progress_callback:
            progress_callback(
                f"Scoring category {cat_idx + 1}/{len(categories)}: {cat['name']}..."
            )

        items_text = "\n".join(
            f"- [{it['id']}] {it['text']}" for it in cat.get("items", [])
        )

        prompt = f"""Score these simulation rubric items based on evidence.

EVIDENCE:
{evidence_summary}

TRANSCRIPT (excerpt):
{transcript_excerpt}

CATEGORY: {cat['name']}
ITEMS:
{items_text}

For each item return: id, score (Yes/Partial/No/N/A), evidence (1 sentence), timestamp.
JSON array only:
[{{"id":"XX1","score":"Yes","evidence":"...","timestamp":"5s"}}]"""

        try:
            raw = vlm_client.ask_text(prompt, model=model, temperature=0.2, max_tokens=4096)
            logger.info("Scoring '%s' raw (%d chars): %s", cat['name'], len(raw), raw[:200])
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            # Try to fix truncated JSON arrays
            raw = _fix_truncated_json(raw)
            scores = json.loads(raw)
            for s in scores:
                if isinstance(s, dict) and "id" in s:
                    all_scores[s["id"]] = s
        except json.JSONDecodeError as e:
            logger.warning("Scoring '%s' returned invalid JSON: %s\nRaw: %s", cat['name'], e, raw[:300])
        except Exception as e:
            logger.warning("Scoring '%s' failed: %s", cat['name'], e)

    scored_rubric = _merge_scores(rubric, all_scores)
    return scored_rubric


def generate_narrative(
    scored_rubric: dict,
    frame_analyses: list[dict],
    character_info: dict,
    transcript: str,
    model: str | None = None,
) -> str:
    """Generate a short narrative summary report."""
    # Build a compact summary of scores
    score_lines = []
    for cat in scored_rubric.get("categories", []):
        for item in cat.get("items", []):
            score = item.get("score", "?")
            score_lines.append(f"[{item.get('id')}] {item['text']}: {score}")
    scores_text = "\n".join(score_lines)

    prompt = f"""You are an emergency medicine simulation debrief facilitator.

Write a concise narrative evaluation report (300-500 words) based on this scored rubric
and video observations.

RUBRIC SCORES:
{scores_text}

TEAM MEMBERS:
{json.dumps(character_info.get('characters', []), indent=2)}

Structure the report as:
1. **Overall Performance Summary** (2-3 sentences)
2. **Strengths** (what was done well)
3. **Areas for Improvement** (specific, actionable feedback)
4. **Key Moments** (notable timestamps or events)
5. **Recommendations** (for next practice session)

Be constructive but honest. This is for educational debriefing, not punitive evaluation.
Reference specific team members by their role labels when possible."""

    try:
        return vlm_client.ask_text(prompt, model=model, temperature=0.4, max_tokens=2000)
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return "Narrative report generation failed. Please review the rubric scores above."


def _fix_truncated_json(raw: str) -> str:
    """Attempt to fix JSON arrays truncated mid-output by the model."""
    raw = raw.strip()
    if not raw:
        return "[]"
    # If it looks like a truncated array, try to close it
    if raw.startswith("[") and not raw.endswith("]"):
        # Find the last complete object (ending with })
        last_brace = raw.rfind("}")
        if last_brace > 0:
            raw = raw[: last_brace + 1] + "]"
    return raw


def _merge_scores(rubric: dict, score_map: dict) -> dict:
    """Merge VLM scores into the rubric structure."""
    scored = json.loads(json.dumps(rubric))  # deep copy
    for cat in scored.get("categories", []):
        for item in cat.get("items", []):
            item_id = item.get("id", "")
            if item_id in score_map:
                item["score"] = score_map[item_id].get("score", "?")
                item["evidence"] = score_map[item_id].get("evidence", "")
                item["timestamp"] = score_map[item_id].get("timestamp", "")
            else:
                item["score"] = "Not Scored"
                item["evidence"] = ""
                item["timestamp"] = ""
    return scored


def _spread_sample(total: int, n: int) -> list[int]:
    """Pick n evenly-spaced indices from range(total)."""
    if n >= total:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]
