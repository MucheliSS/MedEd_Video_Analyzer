"""Manage evaluation rubrics — upload or auto-generate."""

import json
import logging
from pathlib import Path

import vlm_client

logger = logging.getLogger(__name__)

# ── Default ER Simulation Rubric (fallback) ────────────────────────────
DEFAULT_ER_RUBRIC = {
    "title": "Emergency Resuscitation Simulation Rubric",
    "description": "Auto-generated rubric for ER mock resuscitation / sim wars evaluation.",
    "auto_generated": True,
    "categories": [
        {
            "name": "Team Leadership & Communication",
            "items": [
                {"id": "TL1", "text": "Team leader clearly identified and assumed role", "weight": 2},
                {"id": "TL2", "text": "Closed-loop communication used consistently", "weight": 2},
                {"id": "TL3", "text": "Tasks delegated clearly to specific team members", "weight": 2},
                {"id": "TL4", "text": "Situational awareness maintained (verbalized summary)", "weight": 1},
            ],
        },
        {
            "name": "Primary Survey (ABCDE)",
            "items": [
                {"id": "PS1", "text": "Airway assessed and managed appropriately", "weight": 2},
                {"id": "PS2", "text": "Breathing assessed (look/listen/feel, O2 applied)", "weight": 2},
                {"id": "PS3", "text": "Circulation assessed (pulse check, IV access, fluids)", "weight": 2},
                {"id": "PS4", "text": "Disability assessed (GCS/AVPU, pupils)", "weight": 1},
                {"id": "PS5", "text": "Exposure performed appropriately", "weight": 1},
            ],
        },
        {
            "name": "Resuscitation Skills",
            "items": [
                {"id": "RS1", "text": "CPR initiated promptly if indicated (correct rate/depth)", "weight": 2},
                {"id": "RS2", "text": "Defibrillation/cardioversion performed correctly", "weight": 2},
                {"id": "RS3", "text": "Medications administered correctly (drug, dose, route)", "weight": 2},
                {"id": "RS4", "text": "Rhythm recognition and appropriate algorithm followed", "weight": 2},
            ],
        },
        {
            "name": "Situational Awareness & Prioritization",
            "items": [
                {"id": "SA1", "text": "Differential diagnoses considered and verbalized", "weight": 1},
                {"id": "SA2", "text": "Appropriate escalation or consultation requested", "weight": 1},
                {"id": "SA3", "text": "Time awareness maintained (e.g., 2-min rhythm checks)", "weight": 1},
            ],
        },
        {
            "name": "Professionalism & Safety",
            "items": [
                {"id": "PF1", "text": "PPE used appropriately", "weight": 1},
                {"id": "PF2", "text": "Patient safety maintained throughout", "weight": 1},
                {"id": "PF3", "text": "Respectful team interactions observed", "weight": 1},
            ],
        },
    ],
}


def load_rubric_from_json(file_content: str) -> dict:
    """Parse a user-uploaded JSON rubric."""
    rubric = json.loads(file_content)
    _validate_rubric(rubric)
    rubric["auto_generated"] = False
    return rubric


def load_rubric_from_csv(file_content: str) -> dict:
    """Parse a simple CSV rubric (category, id, text, weight)."""
    import csv
    import io

    reader = csv.DictReader(io.StringIO(file_content))
    categories: dict[str, list[dict]] = {}
    for row in reader:
        cat = row.get("category", "General")
        categories.setdefault(cat, []).append({
            "id": row.get("id", ""),
            "text": row.get("text", row.get("item", "")),
            "weight": int(row.get("weight", 1)),
        })

    rubric = {
        "title": "Uploaded Rubric",
        "description": "User-uploaded evaluation rubric.",
        "auto_generated": False,
        "categories": [
            {"name": name, "items": items}
            for name, items in categories.items()
        ],
    }
    _validate_rubric(rubric)
    return rubric


def generate_rubric_from_video(
    scene_descriptions: list[str],
    transcript: str,
    model: str | None = None,
) -> dict:
    """
    Ask the VLM to generate a context-appropriate rubric based on what it
    sees in the video.  Falls back to DEFAULT_ER_RUBRIC on failure.
    """
    scene_summary = "\n".join(
        f"- Frame {i+1}: {desc}" for i, desc in enumerate(scene_descriptions[:10])
    )
    transcript_excerpt = transcript[:2000] if transcript else "(no audio transcript available)"

    prompt = f"""You are an emergency medicine education expert. Based on the following
observations from a simulation video, generate an evaluation rubric in JSON format.

SCENE OBSERVATIONS:
{scene_summary}

AUDIO TRANSCRIPT (excerpt):
{transcript_excerpt}

Generate a JSON rubric with this exact structure:
{{
  "title": "...",
  "description": "...",
  "categories": [
    {{
      "name": "Category Name",
      "items": [
        {{"id": "XX1", "text": "Observable behavior to evaluate", "weight": 1}}
      ]
    }}
  ]
}}

Focus on observable, measurable behaviors relevant to emergency resuscitation simulations.
Include categories for: team communication, clinical skills observed, safety, and decision-making.
Return ONLY valid JSON, no markdown fences or extra text."""

    try:
        raw = vlm_client.ask_text(prompt, model=model, temperature=0.4, max_tokens=3000)
        # Try to extract JSON from the response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        rubric = json.loads(raw)
        rubric["auto_generated"] = True
        _validate_rubric(rubric)
        logger.info("Successfully generated rubric from video content.")
        return rubric
    except Exception as e:
        logger.warning("Rubric generation failed (%s), using default ER rubric.", e)
        return DEFAULT_ER_RUBRIC.copy()


def get_default_rubric() -> dict:
    return DEFAULT_ER_RUBRIC.copy()


def rubric_to_flat_list(rubric: dict) -> list[dict]:
    """Flatten rubric into a list of items with category info."""
    items = []
    for cat in rubric.get("categories", []):
        for item in cat.get("items", []):
            items.append({**item, "category": cat["name"]})
    return items


def _validate_rubric(rubric: dict):
    if "categories" not in rubric:
        raise ValueError("Rubric must have a 'categories' key.")
    for cat in rubric["categories"]:
        if "name" not in cat or "items" not in cat:
            raise ValueError(f"Each category needs 'name' and 'items'. Got: {list(cat.keys())}")
