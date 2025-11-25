# ============================================================
#  Gemini 3 Pro Image Generator (Refactored Full Version)
#  - Supports: Preview 1K / 2K / 4K
#  - Supports: Reference Images
#  - Clean architecture, stable API calls
#  - Works on latest google-genai SDK
# ============================================================

import base64
import uuid
import os
import json
from typing import List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions


# ============================================================
#  CONFIG
# ============================================================

TITLE = "Gemini 画像生成（フルリファクタ版）"

MODEL_OPTIONS = {
    "Gemini 3 Pro Image 1K": "models/gemini-3-pro-image-preview",
    "Gemini 3 Pro Image 2K": "models/gemini-3-pro-image-preview-2k",
    "Gemini 3 Pro Image 4K": "models/gemini-3-pro-image-preview-4k",
}

ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
RESOLUTION_MAP = {
    "1K": "1k",
    "2K": "2k",
    "4K": "4k",
}

DEFAULT_SUFFIX = (
    "((masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus))"
)
NO_TEXT_SUFFIX = (
    "((no text, no watermark, no labels, no subtitles, neutral background))"
)


# ============================================================
#  FUNCTIONS
# ============================================================

def get_api_key() -> str:
    key = (
        st.secrets.get("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    return key.strip()


def require_login() -> None:
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        return

    st.title("ログイン")
    correct_user = st.secrets.get("USERNAME", "mezamashi")
    correct_pass = st.secrets.get("PASSWORD", "mezamashi")

    user = st.text_input("ID")
    pw = st.text_input("PASS", type="password")
    if st.button("ログイン"):
        if user == correct_user and pw == correct_pass:
            st.session_state.auth = True
            st.experimental_rerun()
        else:
            st.error("ID または PASS が違います")

    st.stop()


def extract_image(response) -> Optional[bytes]:
    """
    Gemini API の画像レスポンスは unified format で返される。
    """
    if hasattr(response, "generated_images"):
        # generate_images()
        img = response.generated_images[0].image
        return img.image_bytes

    # generate_content()
    for c in response.candidates:
        for p in c.content.parts:
            if hasattr(p, "inline_data") and p.inline_data:
                return p.inline_data.data

    return None


def render_history():
    if "history" not in st.session_state:
        return

    st.subheader("履歴")
    for entry in st.session_state.history:
        st.image(entry["image"], caption=entry["prompt"], use_container_width=True)
        st.markdown("---")


# ============================================================
#  MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title=TITLE, page_icon="🎨", layout="centered")
    require_login()

    st.title("脳内大喜利・画像生成")

    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API Key が設定されていません")
        return

    # CLIENT
    client = genai.Client(api_key=api_key)

    # UI
    prompt = st.text_area("Prompt", height=150)
    model_label = st.selectbox("モデル", list(MODEL_OPTIONS.keys()))
    aspect = st.selectbox("アスペクト比", ASPECT_RATIOS)
    resolution = st.selectbox("解像度", ["1K", "2K", "4K"])
    refs = st.file_uploader("参考画像 (任意)", accept_multiple_files=True)

    model_name = MODEL_OPTIONS[model_label]
    image_size = RESOLUTION_MAP[resolution]

    if st.button("Generate", type="primary"):

        if not prompt.strip():
            st.warning("プロンプトを入力してください")
            st.stop()

        # 生成プロンプト整形
        full_prompt = f"{prompt}\n{DEFAULT_SUFFIX}\n{NO_TEXT_SUFFIX}"

        # 参考画像構成
        reference_parts = []
        if refs:
            for f in refs:
                reference_parts.append({
                    "inline_data": {
                        "mime_type": f.type,
                        "data": f.getvalue()
                    }
                })

        st.info("🖼️ 生成中... 10〜20秒かかる場合があります")

        try:
            if reference_parts:
                # 参考画像あり → generate_content
                contents = [{
                    "role": "user",
                    "parts": [{"text": full_prompt}, *reference_parts],
                }]

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect),
                    ),
                )
            else:
                # テキストのみ → generate_images（image_size対応）
                response = client.models.generate_images(
                    model=model_name,
                    prompt=full_prompt,
                    config=types.GenerateImagesConfig(
                        aspect_ratio=aspect,
                        image_size=image_size,
                    ),
                )

        except google_exceptions.ResourceExhausted:
            st.error("クォータを超えました")
            st.stop()

        except google_exceptions.GoogleAPICallError as exc:
            st.error(f"API エラー: {exc.message}")
            st.stop()

        except Exception as exc:
            st.error(f"予期しないエラー: {exc}")
            st.stop()

        # 画像抽出
        image_bytes = extract_image(response)
        if not image_bytes:
            st.error("画像が取得できませんでした")
            st.stop()

        # 履歴に保存
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.insert(0, {
            "prompt": prompt,
            "image": image_bytes,
            "model": model_name,
        })

        st.success("生成完了！")
        st.image(image_bytes, use_container_width=True)

    render_history()


# ============================================================

if __name__ == "__main__":
    main()
