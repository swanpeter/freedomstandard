import base64
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from byteplussdkarkruntime import Ark
from basic_setting import logout, require_login
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

I2I_MODEL_ID = "ep-20251208110124-9jp7r"
I2I_MAX_REFERENCES = 14
I2I_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
I2I_DEFAULT_PROMPT = "masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus"


@st.cache_resource
def get_client() -> Ark:
    return Ark(
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
        api_key=os.environ.get("ARK_API_KEY"),
    )


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def derive_filename(image_url: str) -> str:
    parsed = urlparse(image_url)
    base = os.path.basename(parsed.path) or "image.jpeg"
    if "." not in base:
        base = f"{base}.jpeg"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base}"


def download_image(image_url: str) -> Tuple[str, bytes, str]:
    os.makedirs(I2I_OUTPUT_DIR, exist_ok=True)
    filename = derive_filename(image_url)
    output_path = os.path.join(I2I_OUTPUT_DIR, filename)
    response = requests.get(image_url, stream=True, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
    body = bytearray()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                body.extend(chunk)
    return output_path, bytes(body), content_type


def generate_image(
    client: Ark,
    image_data_urls: List[str],
    prompt: str,
) -> Tuple[str, bytes, str]:
    request_args = {
        "model": I2I_MODEL_ID,
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": "2K",
        "stream": False,
        "watermark": False,
    }
    if image_data_urls:
        request_args["image"] = image_data_urls

    response = client.images.generate(**request_args)
    first_item = response.data[0] if getattr(response, "data", None) else None
    image_url = getattr(first_item, "url", None)
    if not image_url:
        raise RuntimeError("レスポンスから画像URLを取得できませんでした。")

    saved_path, image_bytes, mime_type = download_image(image_url)
    return saved_path, image_bytes, mime_type


def main() -> None:
    st.set_page_config(page_title="Seedream 4.5 I2I", layout="wide")
    require_login()
    st.title("Seedream 4.5 I2I (Streamlit)")

    with st.sidebar:
        if st.button("ログアウト"):
            logout()

    st.session_state.setdefault("last_saved_path", None)
    st.session_state.setdefault("last_image_bytes", None)
    st.session_state.setdefault("last_image_name", None)
    st.session_state.setdefault("last_image_mime", "image/jpeg")

    uploaded_files = st.file_uploader(
        "参照画像 (最大14枚)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > I2I_MAX_REFERENCES:
        st.warning(
            f"参照画像は最大{I2I_MAX_REFERENCES}枚までです。先頭{I2I_MAX_REFERENCES}枚のみ使用します。"
        )
        uploaded_files = uploaded_files[:I2I_MAX_REFERENCES]

    prompt = st.text_area("プロンプト", value=I2I_DEFAULT_PROMPT, height=180)
    generate_clicked = st.button("生成", type="primary")

    if generate_clicked:
        if not prompt.strip():
            st.error("プロンプトを入力してください。")
            return

        image_data_urls: List[str] = []
        if uploaded_files:
            for file in uploaded_files:
                image_data_urls.append(image_bytes_to_data_url(file.getvalue()))
        else:
            st.info("参照画像が未選択です。テキストのみで実行します。")

        client = get_client()
        try:
            with st.spinner("生成中..."):
                saved_path, image_bytes, mime_type = generate_image(client, image_data_urls, prompt.strip())
            st.session_state.last_saved_path = saved_path
            st.session_state.last_image_bytes = image_bytes
            st.session_state.last_image_name = os.path.basename(saved_path)
            st.session_state.last_image_mime = mime_type
            st.success("生成が完了しました。")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    last_saved_path = st.session_state.get("last_saved_path")
    if last_saved_path:
        st.image(last_saved_path, caption="生成画像", use_container_width=True)
        image_bytes = st.session_state.get("last_image_bytes")
        file_name = st.session_state.get("last_image_name") or "generated_image.jpeg"
        mime_type = st.session_state.get("last_image_mime") or "image/jpeg"
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            st.download_button(
                "ダウンロード",
                data=bytes(image_bytes),
                file_name=file_name,
                mime=mime_type,
            )


if __name__ == "__main__":
    main()
