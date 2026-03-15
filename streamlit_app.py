from __future__ import annotations

import streamlit as st
import torch

from src.config import Day8Config
from src.device import get_device_info, get_best_device
from src.inference import load_inference_artifacts, predict_single_text_with_probabilities


def get_device() -> torch.device:
    info = get_device_info()
    device = get_best_device()
    print("\n=== CUDA Readiness ===")
    print(f"CUDA available  : {info.get('cuda_available')}")
    print(f"Selected device : {device}")
    if info.get('gpu_name'):
        print(f"GPU name        : {info.get('gpu_name')}")
    return device


@st.cache_resource
def load_runtime(config: Day8Config):
    device = get_device()
    model, tokenizer = load_inference_artifacts(config.model_dir, device)
    return model, tokenizer, device


def main() -> None:
    config = Day8Config()

    st.set_page_config(page_title=config.app_title, layout="centered")
    st.title(config.app_title)
    st.write(config.app_description)

    user_text = st.text_area(config.text_area_label, height=180)

    if st.button(config.button_label):
        if not user_text or not user_text.strip():
            st.warning("Please enter some text before prediction.")
            return

        word_count = len(user_text.strip().split())
        if word_count <= 4:
            st.warning("Not enough context to classify reliably.")
            st.info("Please provide more context (at least 5 words).")
            return

        with st.spinner("Running inference..."):
            try:
                model, tokenizer, device = load_runtime(config)
                prediction = predict_single_text_with_probabilities(
                    text=user_text.strip(),
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=config.max_length,
                )
                fake_prob = prediction["fake_prob"]
                real_prob = prediction["real_prob"]
                confidence = prediction["confidence"]

                if fake_prob >= real_prob:
                    st.success("Prediction: Fake")
                else:
                    st.success("Prediction: Real")
                st.caption(f"Model confidence: {confidence:.2%}")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()