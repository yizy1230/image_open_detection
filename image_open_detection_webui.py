"""Gradio UI for image_open_detection."""

import json
import os

import gradio as gr
import requests

API = os.environ.get("IMAGE_OPEN_DETECTION_API_URL", "http://127.0.0.1:8000") + "/image_open_detection"


def run(image_file, vocab_tier, score_thr):
    if not image_file:
        return "Please upload an image.", ""
    try:
        resp = requests.post(
            API,
            json={"image_path": image_file, "vocab_tier": vocab_tier, "score_thr": score_thr},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        return f"API Error: {exc}", ""
    status = (
        f"instances={data.get('num_instances')} "
        f"subjects={data.get('subjects')} env={data.get('environment')}"
    )
    return status, json.dumps(data, ensure_ascii=False, indent=2)


with gr.Blocks(title="Image Open Detection") as demo:
    inp = gr.Image(type="filepath", label="Image")
    tier = gr.Dropdown(["core", "extended", "all"], value="core", label="vocab_tier")
    thr = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="score_thr")
    btn = gr.Button("Detect")
    status = gr.Textbox(label="Status")
    raw = gr.Code(language="json")
    btn.click(fn=run, inputs=[inp, tier, thr], outputs=[status, raw])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
