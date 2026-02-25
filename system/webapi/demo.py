from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import gradio as gr
import httpx

UI_VERSION = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-utc")


CSS = """
:root {
  --bg-a: #120f16;
  --bg-b: #21152b;
  --line: rgba(255,255,255,0.16);
  --text: #f8f5ff;
  --muted: #cabfe0;
  --accent: #ff8a5a;
}
.gradio-container {
  font-family: "Noto Sans KR", "Pretendard", "SUIT", sans-serif !important;
  color: var(--text);
  background:
    radial-gradient(1200px 500px at 10% 0%, rgba(255, 192, 220, 0.36), transparent 60%),
    radial-gradient(900px 420px at 90% 5%, rgba(255, 213, 233, 0.30), transparent 58%),
    linear-gradient(180deg, rgba(255, 240, 247, 0.68), rgba(255, 247, 251, 0.56));
}
.vn-root { max-width: 1680px; margin: 0 auto; }
.vn-head { margin: 6px 0 12px 0; }
.vn-title {
  font-size: 34px; font-weight: 800; margin: 0; letter-spacing: 0.02em;
  color: #ffffff; text-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
.vn-sub { margin: 4px 0 0 0; color: #fff6fb; font-size: 14px; text-shadow: 0 1px 8px rgba(0,0,0,0.35); }

.stage-wrap {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
  background-color: #0d0d12;
  min-height: clamp(560px, 76vh, 820px);
}
.stage-wrap::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(6,6,10,0.28), rgba(6,6,10,0.58));
  z-index: 2;
  pointer-events: none;
}
.stage-badge {
  position: absolute;
  top: 14px;
  left: 14px;
  z-index: 15;
  border: 1px solid rgba(255,255,255,0.32);
  border-radius: 999px;
  padding: 6px 12px;
  background: rgba(0,0,0,0.26);
  color: #ffffff;
  font-size: 12px;
  font-weight: 700;
}
.character-img {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 0;
  width: min(42vw, 460px);
  max-width: 460px;
  z-index: 6;
  pointer-events: none;
}
.character-img img.vn-portrait {
  width: 100%;
  height: auto;
  object-fit: contain !important;
  object-position: center bottom !important;
  filter: drop-shadow(0 18px 22px rgba(0,0,0,0.45));
}

.dialog-panel {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: min(92%, 1100px);
  bottom: 72px;
  z-index: 20;
  background: rgba(7, 7, 10, 0.84);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px;
  backdrop-filter: blur(6px);
}
.speaker-tag {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 10px;
  margin-bottom: 6px;
  font-size: 13px;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.3);
  background: rgba(255,138,90,0.28);
  color: #fff9ef !important;
}
.vn-narr { color: #ffffff !important; margin-bottom: 6px; font-size: 16px; text-shadow: 0 1px 6px rgba(0,0,0,0.55); opacity: 1 !important; }
.vn-line { color: #ffffff !important; font-size: 24px; line-height: 1.55; text-shadow: 0 1px 6px rgba(0,0,0,0.65); opacity: 1 !important; }
.vn-ja { color: #d6d0e6; margin-top: 6px; font-size: 13px; }

#vn-stage-host { position: relative; }
#vn-input-wrap {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 16px;
  z-index: 28;
  width: min(96%, 1320px);
  background: rgba(7,7,10,0.88);
  border: 1px solid rgba(255,255,255,0.30);
  border-radius: 14px;
  padding: 8px;
  backdrop-filter: blur(4px);
}
#vn-input-row {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  gap: 8px;
  align-items: center;
}
#vn-user-input {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}
#vn-run-btn {
  flex: 0 0 132px !important;
  max-width: 132px !important;
  min-width: 132px !important;
}
.stage-input-row textarea {
  background: rgba(9,9,14,0.86) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.42) !important;
  border-radius: 12px !important;
}
.stage-input-row button {
  height: 48px !important;
}
.scene-reset-wrap {
  width: min(96%, 1320px);
  margin: 0 auto 8px auto;
}
.scene-reset-wrap button {
  width: 100%;
  height: 48px !important;
}
.dev-face-wrap { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.dev-face-wrap img { width: 92px; height: 92px; object-fit: cover; border-radius: 10px; border: 1px solid var(--line); }
.dev-face-label { font-size: 12px; color: var(--muted); margin-top: 2px; }
.stealth-audio {
  height: 2px !important;
  min-height: 2px !important;
  opacity: 0.01 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
}
.stealth-audio audio {
  height: 1px !important;
}
@media (max-width: 940px) {
  .stage-wrap { min-height: 480px; }
  .character-img {
    left: 50%;
    transform: translateX(-50%);
    width: min(60vw, 320px);
  }
  .dialog-panel {
    width: calc(100% - 24px);
    bottom: 56px;
  }
  .vn-line { font-size: 18px; }
  #vn-input-wrap {
    width: calc(100% - 20px);
    bottom: 10px;
  }
  #vn-input-row {
    flex-direction: column !important;
  }
  #vn-run-btn {
    max-width: none !important;
    width: 100% !important;
    min-width: 0 !important;
  }
  .scene-reset-wrap {
    width: calc(100% - 20px);
  }
}
"""


def _render_dialog_inner(record: dict[str, Any] | str | None) -> str:
    if not record:
        return '<div class="speaker-tag">SAYA</div><div class="vn-line">...</div>'

    narr = html.escape(record.get("narration", ""))
    dko = html.escape(record.get("dialogue_ko", ""))
    dja = html.escape(record.get("dialogue_ja", ""))
    if not dko:
        dko = "..."

    parts = ['<div class="speaker-tag">SAYA</div>']
    if narr:
        parts.append(f'<div class="vn-narr">{narr}</div>')
    parts.append(f'<div class="vn-line">"{dko}"</div>')
    return "".join(parts)


def _render_stage(record: dict[str, Any] | None, bg_b64: str, char_b64: str) -> str:
    dialog = _render_dialog_inner(record)
    return (
        '<div style="position:relative;border:1px solid rgba(255,255,255,0.16);'
        'border-radius:18px;overflow:hidden;min-height:clamp(580px,78vh,860px);'
        f"background-image:url('data:image/jpeg;base64,{bg_b64}');background-size:cover;background-position:center center;"
        '">'
        '<div style="position:absolute;inset:0;background:linear-gradient(180deg,rgba(8,8,12,0.12),rgba(8,8,12,0.44));z-index:2;"></div>'
        '<div style="position:absolute;top:14px;left:14px;z-index:15;border:1px solid rgba(255,255,255,0.32);'
        'border-radius:999px;padding:6px 12px;background:rgba(0,0,0,0.46);font-size:12px;font-weight:700;color:#ffffff;">'
        'Character: 사야 (SAYA)</div>'
        '<div style="position:absolute;left:50%;transform:translateX(-50%);bottom:0;z-index:6;pointer-events:none;'
        'height:100%;width:100%;display:flex;justify-content:center;align-items:flex-end;">'
        '<div style="position:relative;height:100%;display:flex;align-items:flex-end;justify-content:center;">'
        f"<img src='data:image/png;base64,{char_b64}' alt='SAYA neutral' "
        "style='position:relative;height:120%;width:auto;max-height:186%;max-width:min(104vw,1190px);"
        "transform-origin:center bottom;transform:translateY(24%);"
        "object-fit:contain;object-position:center bottom;image-rendering:auto;"
        "filter:drop-shadow(0 22px 26px rgba(0,0,0,0.45));'/>"
        "</div>"
        "</div>"
        '<div style="position:absolute;left:50%;transform:translateX(-50%);bottom:56px;z-index:20;'
        'width:min(96%,1500px);background:rgba(7,7,10,0.88);border:1px solid rgba(255,255,255,0.30);'
        'border-radius:14px;padding:22px 26px;min-height:190px;backdrop-filter:blur(4px);">'
        f"{dialog}"
        "</div>"
        "</div>"
    )


def _normalize_emotion(raw: Any) -> dict[str, int]:
    out = {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
    if not isinstance(raw, dict):
        return out
    for key in out:
        out[key] = 1 if int(raw.get(key, 0)) > 0 else 0
    if sum(out.values()) == 0:
        out["neutral"] = 1
    if sum(out.values()) > 1:
        # Ensure one-hot by priority: angry > sad > happy > neutral.
        if out["angry"]:
            return {"neutral": 0, "sad": 0, "happy": 0, "angry": 1}
        if out["sad"]:
            return {"neutral": 0, "sad": 1, "happy": 0, "angry": 0}
        if out["happy"]:
            return {"neutral": 0, "sad": 0, "happy": 1, "angry": 0}
        return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
    return out


def _pick_char_b64(emotion: dict[str, int], char_b64_map: dict[str, str]) -> str:
    if emotion.get("angry", 0) == 1:
        return char_b64_map["angry"]
    if emotion.get("sad", 0) == 1:
        return char_b64_map["sad"]
    if emotion.get("happy", 0) == 1:
        return char_b64_map["happy"]
    return char_b64_map["neutral"]


def _render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "no turns yet"
    lines = []
    for i, h in enumerate(history[-20:], start=max(1, len(history) - 19)):
        user = h.get("user", "")
        dko = h.get("dialogue_ko", "")
        lines.append(f"{i:02d}. U: {user}\n    A: {dko}")
    return "\n".join(lines)


def build_demo(base_url: str) -> gr.Blocks:
    asset_neutral = Path(__file__).resolve().parent / "assets" / "neutral_full.png"
    asset_sad = Path(__file__).resolve().parent / "assets" / "crying_full.png"
    asset_happy = Path(__file__).resolve().parent / "assets" / "smile_full.png"
    asset_angry = Path(__file__).resolve().parent / "assets" / "annoyed_full.png"
    dev_face_path = Path(__file__).resolve().parent / "assets" / "face_neutral.png"
    bg_path = Path(__file__).resolve().parent / "assets" / "background.jpg"
    for p in [asset_neutral, asset_sad, asset_happy, asset_angry]:
        if not p.exists():
            raise FileNotFoundError(f"Required asset not found: {p}")
    if not dev_face_path.exists():
        raise FileNotFoundError(f"Required asset not found: {dev_face_path}")
    if not bg_path.exists():
        raise FileNotFoundError(f"Required asset not found: {bg_path}")
    char_b64_map = {
        "neutral": base64.b64encode(asset_neutral.read_bytes()).decode("ascii"),
        "sad": base64.b64encode(asset_sad.read_bytes()).decode("ascii"),
        "happy": base64.b64encode(asset_happy.read_bytes()).decode("ascii"),
        "angry": base64.b64encode(asset_angry.read_bytes()).decode("ascii"),
    }
    dev_face_b64 = base64.b64encode(dev_face_path.read_bytes()).decode("ascii")
    bg_b64 = base64.b64encode(bg_path.read_bytes()).decode("ascii")
    dev_face_html = (
        '<div class="dev-face-wrap">'
        f'<img src="data:image/png;base64,{dev_face_b64}" alt="face_neutral"/>'
        '<div><div class="dev-face-label">face_neutral</div></div>'
        "</div>"
    )

    def mainloop_turn(user_text: str, history: list[dict[str, Any]]):
        payload = {"text_ko": user_text}
        with httpx.Client(timeout=240.0) as client:
            res = client.post(f"{base_url}/api/main-loop", json=payload)
            res.raise_for_status()
            body = res.json()

        rec = {
            "user": user_text,
            "narration": body.get("narration", ""),
            "dialogue_ko": body.get("dialogue_ko", ""),
            "dialogue_ja": body.get("dialogue_ja", ""),
            "emotion": _normalize_emotion(body.get("emotion")),
        }
        new_history = (history or []) + [rec]
        wav_path = body.get("wav_path", None)
        char_b64 = _pick_char_b64(rec["emotion"], char_b64_map)
        return (
            _render_stage(rec, bg_b64=bg_b64, char_b64=char_b64),
            wav_path,
            _render_history(new_history),
            body.get("rp_text", ""),
            body.get("narration", ""),
            body.get("dialogue_ko", ""),
            body.get("dialogue_ja", ""),
            body.get("wav_path", ""),
            new_history,
        )

    def reset_scene():
        return (
            _render_stage(None, bg_b64=bg_b64, char_b64=char_b64_map["neutral"]),
            gr.skip(),
            "no turns yet",
            "",
            "",
            "",
            "",
            "",
            [],
        )

    def chat_api(user_text: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{base_url}/api/chat", json={"text": user_text})
            res.raise_for_status()
            return res.json()["response"]

    def trans_api(text_ko: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{base_url}/api/translate", json={"text_ko": text_ko})
            res.raise_for_status()
            return res.json()["text_ja"]

    def tts_api(text_ja: str):
        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{base_url}/api/tts", json={"text_ja": text_ja})
            res.raise_for_status()
            wav = res.json()["wav_path"]
            return wav, wav

    with gr.Blocks(title="SAYA MainLoop Demo") as demo:
        history_state = gr.State([])

        gr.HTML(
            f"<style>{CSS}</style>"
            '<div class="vn-root vn-head">'
            f'<h1 class="vn-title">SAYA / MainLoop UI v{UI_VERSION}</h1>'
            f'<p class="vn-sub">REST backend: {html.escape(base_url)} | pipeline: input -> LLM -> parse -> translate -> TTS</p>'
            "</div>"
        )

        with gr.Group(elem_classes=["vn-root"], elem_id="vn-stage-host"):
            stage_html = gr.HTML(_render_stage(None, bg_b64=bg_b64, char_b64=char_b64_map["neutral"]))

            with gr.Group(elem_id="vn-input-wrap"):
                with gr.Row(elem_id="vn-input-row"):
                    user_input = gr.Textbox(
                        show_label=False,
                        lines=1,
                        placeholder='대화창 안에서 입력: 예) "사야, 오늘 기분 어때?"',
                        container=False,
                        elem_id="vn-user-input",
                    )
                    run_btn = gr.Button("대화 진행", variant="primary", elem_id="vn-run-btn")
            # Keep mounted for autoplay trigger, but visually hide via CSS.
            audio_hidden = gr.Audio(
                label="Voice",
                type="filepath",
                autoplay=True,
                visible=True,
                elem_classes=["stealth-audio"],
            )

        with gr.Group(elem_classes=["vn-root", "scene-reset-wrap"]):
            reset_btn = gr.Button("씬 초기화")

        with gr.Accordion("Transcript / Debug", open=False):
            transcript = gr.Textbox(label="Transcript", lines=12, value="no turns yet")
            dbg_rp = gr.Textbox(label="RP Raw")
            dbg_narr = gr.Textbox(label="Narration")
            dbg_dko = gr.Textbox(label="Dialogue (KO)")
            dbg_dja = gr.Textbox(label="Dialogue (JA)")
            dbg_wav = gr.Textbox(label="WAV Path")

        with gr.Accordion("Dev/Test Tools", open=False):
            gr.HTML(dev_face_html)
            with gr.Tab("Chat only"):
                c_in = gr.Textbox(label="KO Input")
                c_out = gr.Textbox(label="RP Output")
                c_btn = gr.Button("POST /api/chat")
                c_btn.click(chat_api, inputs=c_in, outputs=c_out)

            with gr.Tab("Translate only"):
                tr_in = gr.Textbox(label="KO Dialogue")
                tr_out = gr.Textbox(label="JA Dialogue")
                tr_btn = gr.Button("POST /api/translate")
                tr_btn.click(trans_api, inputs=tr_in, outputs=tr_out)

            with gr.Tab("TTS only"):
                ts_in = gr.Textbox(label="JA Text")
                ts_out = gr.Textbox(label="WAV Path")
                ts_audio = gr.Audio(label="Audio", type="filepath", autoplay=True)
                ts_btn = gr.Button("POST /api/tts")
                ts_btn.click(tts_api, inputs=ts_in, outputs=[ts_out, ts_audio])

        run_btn.click(
            mainloop_turn,
            inputs=[user_input, history_state],
            outputs=[stage_html, audio_hidden, transcript, dbg_rp, dbg_narr, dbg_dko, dbg_dja, dbg_wav, history_state],
            show_progress="hidden",
        )
        user_input.submit(
            mainloop_turn,
            inputs=[user_input, history_state],
            outputs=[stage_html, audio_hidden, transcript, dbg_rp, dbg_narr, dbg_dko, dbg_dja, dbg_wav, history_state],
            show_progress="hidden",
        )

        reset_btn.click(
            reset_scene,
            outputs=[stage_html, audio_hidden, transcript, dbg_rp, dbg_narr, dbg_dko, dbg_dja, dbg_wav, history_state],
            show_progress="hidden",
        )

    return demo
