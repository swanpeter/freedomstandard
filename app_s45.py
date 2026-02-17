import base64
import os
import time
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import requests
import streamlit as st
from byteplussdkarkruntime import Ark
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

I2I_MODEL_ID = "ep-20251208110124-9jp7r"
I2I_MAX_REFERENCES = 14
I2I_DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "outputs")
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


def download_image(image_url: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = derive_filename(image_url)
    output_path = os.path.join(save_dir, filename)
    response = requests.get(image_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    return output_path


def generate_image(
    client: Ark,
    image_data_urls: List[str],
    prompt: str,
    save_dir: str,
    log,
) -> Optional[str]:
    log("BytePlusへリクエスト送信中...")
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

    log(f"生成完了: {image_url}")
    saved_path = download_image(image_url, save_dir)
    log(f"画像を保存しました: {saved_path}")
    return saved_path


def main() -> None:
    st.set_page_config(page_title="Seedream 4.5 I2I", layout="wide")
    st.title("Seedream 4.5 I2I (Streamlit)")

    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("last_saved_path", None)

    log_placeholder = st.empty()

    def log(message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.logs.append(f"[{timestamp}] {message}")
        log_placeholder.text("\n".join(st.session_state.logs))

    with st.sidebar:
        st.subheader("設定")
        save_dir = st.text_input("保存先", value=I2I_DEFAULT_SAVE_DIR)
        st.caption("保存先はローカルディスクに保存されます。")

    left, right = st.columns([2, 3])

    with left:
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

    with right:
        st.subheader("ログ")
        log_placeholder.text("\n".join(st.session_state.logs))
        st.subheader("生成結果")
        if st.session_state.last_saved_path:
            st.image(st.session_state.last_saved_path, caption="保存済み画像", use_container_width=True)

    if generate_clicked:
        st.session_state.logs = []
        log("生成を開始します...")

        if not prompt.strip():
            st.error("プロンプトを入力してください。")
            return
        if not save_dir.strip():
            st.error("保存先ディレクトリを入力してください。")
            return

        image_data_urls: List[str] = []
        if uploaded_files:
            log(f"{len(uploaded_files)}枚の参照画像をエンコード中...")
            for file in uploaded_files:
                image_data_urls.append(image_bytes_to_data_url(file.getvalue()))
        else:
            st.info("参照画像が未選択です。テキストのみで実行します。")

        client = get_client()
        try:
            with st.spinner("生成中..."):
                saved_path = generate_image(client, image_data_urls, prompt.strip(), save_dir.strip(), log)
            st.session_state.last_saved_path = saved_path
            if saved_path:
                st.success("生成が完了しました。")
        except Exception as exc:  # noqa: BLE001
            log(f"エラー: {exc}")
            st.error(str(exc))


if __name__ == "__main__":
    main()
