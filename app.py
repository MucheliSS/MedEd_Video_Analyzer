"""
ER Simulation Video Analyzer — Streamlit App
Analyzes emergency room simulation videos against evaluation rubrics
using a local VLM (Gemma 4 via LM Studio) for complete privacy.
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
import vlm_client
import video_processor
import rubric_manager
import analyzer
import report_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ER Sim Video Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────
for key, default in {
    "rubric": None,
    "rubric_source": None,
    "frames": [],
    "video_meta": {},
    "transcript": "",
    "character_info": {},
    "frame_analyses": [],
    "scored_rubric": None,
    "narrative": "",
    "report_dir": None,
    "analysis_step": 0,     # 0=idle, 1=extracting, 2=characters, 3=analyzing, 4=scoring, 5=done
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ER Sim Video Analyzer")
    st.caption("Privacy-first | Local VLM via LM Studio")

    st.divider()

    # Model selection
    st.subheader("Model Settings")
    available_models = vlm_client.list_models()
    if available_models:
        selected_model = st.selectbox(
            "VLM Model",
            available_models,
            help="Models currently loaded in LM Studio",
        )
    else:
        selected_model = st.text_input(
            "Model name",
            value=config.DEFAULT_MODEL,
            help="LM Studio not reachable or no models loaded. Enter model name manually.",
        )
        st.warning("Could not connect to LM Studio. Make sure it's running on localhost:1234.")

    st.divider()

    # Frame extraction settings
    st.subheader("Frame Extraction")
    frame_interval = st.slider(
        "Seconds between frames", 1, 15, config.FRAME_INTERVAL_SEC,
        help="Lower = more frames = more detail but slower analysis",
    )
    max_frames = st.slider(
        "Max frames to extract", 10, 120, config.MAX_FRAMES,
    )
    enable_audio = st.checkbox("Transcribe audio (requires Whisper + ffmpeg)", value=True)

    st.divider()
    st.markdown(
        "**All processing is local.**  \n"
        "Video frames are analyzed by your local VLM.  \n"
        "No data leaves your machine."
    )


# ── Main area ─────────────────────────────────────────────────────────
st.header("Emergency Room Simulation Video Analyzer")

col_video, col_rubric = st.columns([1, 1])

# ── Video upload ──────────────────────────────────────────────────────
with col_video:
    st.subheader("1. Upload Simulation Video")
    uploaded_video = st.file_uploader(
        "Select video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload the recorded simulation video",
    )
    if uploaded_video:
        # Save to uploads dir
        video_path = config.UPLOADS_DIR / uploaded_video.name
        video_path.write_bytes(uploaded_video.getvalue())
        st.video(str(video_path))
        meta = video_processor.get_video_metadata(str(video_path))
        st.session_state.video_meta = meta
        st.caption(
            f"Duration: {meta['duration_sec']}s | "
            f"Resolution: {meta['resolution']} | "
            f"FPS: {meta['fps']:.0f}"
        )

# ── Rubric upload / generation ────────────────────────────────────────
with col_rubric:
    st.subheader("2. Evaluation Rubric")
    rubric_mode = st.radio(
        "Rubric source",
        ["Upload rubric", "Auto-generate from video", "Use default ER rubric"],
        help="Upload your own rubric or let the model generate one based on video content.",
    )

    if rubric_mode == "Upload rubric":
        rubric_file = st.file_uploader(
            "Upload rubric (JSON or CSV)",
            type=["json", "csv"],
        )
        if rubric_file:
            try:
                content = rubric_file.getvalue().decode("utf-8")
                if rubric_file.name.endswith(".json"):
                    st.session_state.rubric = rubric_manager.load_rubric_from_json(content)
                else:
                    st.session_state.rubric = rubric_manager.load_rubric_from_csv(content)
                st.session_state.rubric_source = "user_upload"
                st.success(f"Rubric loaded: {st.session_state.rubric.get('title', 'Untitled')}")
            except Exception as e:
                st.error(f"Failed to parse rubric: {e}")

    elif rubric_mode == "Auto-generate from video":
        st.info(
            "The rubric will be **auto-generated** by the AI model after analyzing "
            "initial frames. The report will clearly note that the rubric was not "
            "human-authored."
        )
        st.session_state.rubric_source = "auto_generate"

    else:  # default
        st.session_state.rubric = rubric_manager.get_default_rubric()
        st.session_state.rubric_source = "default"
        st.info("Using built-in ER resuscitation rubric.")

    # Preview current rubric
    if st.session_state.rubric and st.session_state.rubric_source != "auto_generate":
        with st.expander("Preview Rubric", expanded=False):
            for cat in st.session_state.rubric.get("categories", []):
                st.markdown(f"**{cat['name']}**")
                for item in cat["items"]:
                    st.markdown(f"- `{item['id']}` {item['text']} (weight: {item['weight']})")

# ── Download sample rubric ────────────────────────────────────────────
with col_rubric:
    with st.expander("Download sample rubric templates"):
        sample_json = json.dumps(rubric_manager.DEFAULT_ER_RUBRIC, indent=2)
        st.download_button(
            "Download sample JSON rubric",
            data=sample_json,
            file_name="sample_rubric.json",
            mime="application/json",
        )
        # CSV sample
        csv_lines = ["category,id,text,weight"]
        for cat in rubric_manager.DEFAULT_ER_RUBRIC["categories"]:
            for item in cat["items"]:
                csv_lines.append(f'"{cat["name"]}",{item["id"]},"{item["text"]}",{item["weight"]}')
        st.download_button(
            "Download sample CSV rubric",
            data="\n".join(csv_lines),
            file_name="sample_rubric.csv",
            mime="text/csv",
        )

st.divider()

# ── Analysis ──────────────────────────────────────────────────────────
st.subheader("3. Run Analysis")

can_run = uploaded_video is not None and (
    st.session_state.rubric is not None or st.session_state.rubric_source == "auto_generate"
)

if not can_run:
    st.info("Upload a video and select a rubric option to begin analysis.")

if can_run and st.button("Analyze Video", type="primary", use_container_width=True):
    video_path = config.UPLOADS_DIR / uploaded_video.name
    progress = st.progress(0, text="Starting analysis...")
    status = st.status("Analysis Pipeline", expanded=True)

    def update_progress(pct, text):
        progress.progress(pct, text=text)

    try:
        # Step 1: Extract frames
        with status:
            st.write("Extracting frames from video...")
        update_progress(5, "Extracting frames...")
        frames = video_processor.extract_frames(
            video_path,
            interval_sec=frame_interval,
            max_frames=max_frames,
        )
        st.session_state.frames = frames
        with status:
            st.write(f"Extracted {len(frames)} frames.")

        # Step 1b: Audio transcription
        transcript = ""
        if enable_audio:
            with status:
                st.write("Extracting and transcribing audio...")
            update_progress(15, "Transcribing audio...")
            audio_path = video_processor.extract_audio(video_path)
            if audio_path:
                transcript = video_processor.transcribe_audio(audio_path)
                with status:
                    st.write(f"Transcript: {len(transcript)} characters.")
            else:
                with status:
                    st.write("No usable audio track found.")
        st.session_state.transcript = transcript

        # Step 2: Identify characters
        update_progress(25, "Identifying team members...")
        with status:
            st.write("Identifying team members across frames...")
        char_info = analyzer.identify_characters(frames, model=selected_model)
        st.session_state.character_info = char_info
        n_chars = len(char_info.get("characters", []))
        with status:
            st.write(f"Identified {n_chars} team member(s).")

        # Step 2b: Auto-generate rubric if needed
        if st.session_state.rubric_source == "auto_generate":
            update_progress(35, "Generating rubric from video content...")
            with status:
                st.write("Auto-generating evaluation rubric...")
            # Get quick scene descriptions for rubric generation
            scene_descs = []
            for fa in (st.session_state.frame_analyses or []):
                scene_descs.append(str(fa.get("actions", [])))
            # If no analyses yet, get them from character info
            if not scene_descs:
                scene_descs = [char_info.get("scene_description", "")]
            st.session_state.rubric = rubric_manager.generate_rubric_from_video(
                scene_descs, transcript, model=selected_model
            )
            with status:
                st.write(f"Generated rubric: {st.session_state.rubric.get('title', '?')}")

        # Step 3: Analyze frames in batches
        update_progress(40, "Analyzing clinical actions in frames...")
        with status:
            st.write("Analyzing frames for clinical actions...")

        def batch_progress(msg):
            with status:
                st.write(msg)

        frame_analyses = analyzer.analyze_frames_in_batches(
            frames, char_info, model=selected_model, progress_callback=batch_progress,
        )
        st.session_state.frame_analyses = frame_analyses
        update_progress(70, "Frame analysis complete.")

        # Step 4: Score rubric
        update_progress(75, "Scoring rubric items...")
        with status:
            st.write("Scoring rubric against observed evidence...")
        scored_rubric = analyzer.score_rubric(
            st.session_state.rubric,
            frame_analyses,
            char_info,
            transcript,
            model=selected_model,
        )
        st.session_state.scored_rubric = scored_rubric

        # Step 5: Generate narrative
        update_progress(90, "Generating narrative report...")
        with status:
            st.write("Writing narrative evaluation report...")
        narrative = analyzer.generate_narrative(
            scored_rubric, frame_analyses, char_info, transcript, model=selected_model,
        )
        st.session_state.narrative = narrative

        # Step 6: Save report
        update_progress(95, "Saving report...")
        report_dir = report_generator.save_report(
            uploaded_video.name,
            scored_rubric,
            narrative,
            char_info,
            st.session_state.video_meta,
            transcript,
        )
        st.session_state.report_dir = report_dir

        update_progress(100, "Analysis complete!")
        status.update(label="Analysis Complete", state="complete")

    except Exception as e:
        status.update(label="Analysis Failed", state="error")
        st.error(f"Analysis failed: {e}")
        logger.exception("Analysis pipeline error")


# ── Results display ───────────────────────────────────────────────────
if st.session_state.scored_rubric:
    st.divider()
    st.header("Results")

    # Transparency banner
    if st.session_state.scored_rubric.get("auto_generated"):
        st.warning(
            "**Transparency Notice:** The evaluation rubric was **auto-generated** by the AI model "
            "based on video content. It was NOT provided by a human evaluator. Review rubric items "
            "for appropriateness before using these scores for any formal assessment."
        )

    tab_rubric, tab_narrative, tab_characters, tab_timeline, tab_export = st.tabs(
        ["Rubric Scores", "Narrative Report", "Team Members", "Timeline", "Export"]
    )

    # ── Tab: Rubric Scores ────────────────────────────────────────────
    with tab_rubric:
        total_yes = total_partial = total_no = total_na = 0
        for cat in st.session_state.scored_rubric.get("categories", []):
            st.markdown(f"### {cat['name']}")
            rows = []
            for item in cat.get("items", []):
                score = item.get("score", "?")
                if score == "Yes":
                    total_yes += 1
                    icon = "&#9989;"
                elif score == "Partial":
                    total_partial += 1
                    icon = "&#11036;"
                elif score == "No":
                    total_no += 1
                    icon = "&#10060;"
                else:
                    total_na += 1
                    icon = "&#11035;"
                rows.append({
                    "": icon,
                    "ID": item.get("id", ""),
                    "Item": item["text"],
                    "Score": score,
                    "Evidence": item.get("evidence", ""),
                    "Timestamp": item.get("timestamp", ""),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        # Summary scores
        st.divider()
        scored_total = total_yes + total_partial + total_no
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Yes", total_yes)
        col2.metric("Partial", total_partial)
        col3.metric("No", total_no)
        col4.metric("N/A", total_na)
        if scored_total > 0:
            pct = round((total_yes + 0.5 * total_partial) / scored_total * 100, 1)
            col5.metric("Overall Score", f"{pct}%")

    # ── Tab: Narrative Report ─────────────────────────────────────────
    with tab_narrative:
        st.markdown(st.session_state.narrative)

    # ── Tab: Team Members ─────────────────────────────────────────────
    with tab_characters:
        chars = st.session_state.character_info.get("characters", [])
        if chars:
            for c in chars:
                st.markdown(
                    f"**{c.get('label', '?')}**  \n"
                    f"Visual ID: {c.get('visual_id', '?')}  \n"
                    f"Role: {c.get('apparent_role', '?')}"
                )
                st.divider()
        else:
            st.info("No team members identified.")

        scene = st.session_state.character_info.get("scene_description", "")
        if scene:
            st.markdown(f"**Scene:** {scene}")

    # ── Tab: Timeline ─────────────────────────────────────────────────
    with tab_timeline:
        if st.session_state.frame_analyses:
            for fa in st.session_state.frame_analyses:
                ts = fa.get("timestamp", "?")
                actions = ", ".join(fa.get("actions", []))
                chars_involved = ", ".join(fa.get("characters_involved", []))
                comm = fa.get("communication_observed") or ""
                concerns = fa.get("concerns") or ""

                with st.expander(f"**{ts}s** — {actions[:80]}"):
                    st.markdown(f"**Actions:** {actions}")
                    if chars_involved:
                        st.markdown(f"**Team members:** {chars_involved}")
                    if comm:
                        st.markdown(f"**Communication:** {comm}")
                    if concerns:
                        st.markdown(f"**Concerns:** {concerns}")
        else:
            st.info("No frame analyses available.")

        # Transcript
        if st.session_state.transcript:
            st.divider()
            st.markdown("### Audio Transcript")
            st.text_area(
                "Transcript",
                st.session_state.transcript,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

    # ── Tab: Export ───────────────────────────────────────────────────
    with tab_export:
        if st.session_state.report_dir:
            report_dir = Path(st.session_state.report_dir)
            st.success(f"Reports saved to: `{report_dir}`")

            # JSON download
            json_path = report_dir / "report.json"
            if json_path.exists():
                st.download_button(
                    "Download Full Report (JSON)",
                    data=json_path.read_text(encoding="utf-8"),
                    file_name="evaluation_report.json",
                    mime="application/json",
                )

            # Text download
            txt_path = report_dir / "report.txt"
            if txt_path.exists():
                st.download_button(
                    "Download Report (Text)",
                    data=txt_path.read_text(encoding="utf-8"),
                    file_name="evaluation_report.txt",
                    mime="text/plain",
                )
