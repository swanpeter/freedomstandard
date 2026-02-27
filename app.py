import base64
import datetime
import html
import io
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
from PIL import Image
import requests

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except ImportError:
    StreamlitSecretNotFoundError = Exception

try:
    from google import genai
    from google.api_core import exceptions as google_exceptions
    from google.genai import types
    from google.cloud import storage
    from google.oauth2 import service_account
    import vertexai
    from vertexai.preview.vision_models import Image as VertexImage
    from vertexai.preview.vision_models import ImageGenerationModel
except ImportError:
    st.error(
        "必要なライブラリが不足しています。`pip install -r requirements.txt` を実行してください。"
    )
    st.stop()

def get_secret_value(key: str) -> Optional[str]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None
    try:
        return secrets_obj[key]
    except (KeyError, TypeError, StreamlitSecretNotFoundError):
        pass
    get_method = getattr(secrets_obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except Exception:
            return None
    return None


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


TITLE = "Gemini 画像生成"

MODEL_NAME = "models/gemini-3.1-flash-image-preview"
##MODEL_NAME = "models/gemini-3-pro-image-preview"
IMAGEN_UPSCALE_MODEL = "imagen-4.0-upscale-preview"
IMAGE_ASPECT_RATIO = "16:9"
IMAGE_ASPECT_RATIO_OPTIONS = ("16:9", "9:16", "1:1")
DEFAULT_PROMPT_SUFFIX = (
    "((masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus))"
)
NO_TEXT_TOGGLE_SUFFIX = (
    ""
)

DEFAULT_GEMINI_API_KEY = (
    get_secret_value("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
)


def _is_truthy(value: Optional[object]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def is_gcs_upload_enabled() -> bool:
    raw = get_secret_value("ENABLE_GCS_UPLOAD")
    if raw is None:
        raw = os.getenv("ENABLE_GCS_UPLOAD")
    return _is_truthy(raw)


def _normalize_credential(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_secret_auth_credentials() -> Tuple[Optional[str], Optional[str]]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None, None
    except Exception:
        return None, None

    auth_section: Optional[Dict[str, Any]] = None
    if isinstance(secrets_obj, dict):
        auth_section = secrets_obj.get("auth")
    else:
        auth_section = getattr(secrets_obj, "get", lambda _key, _default=None: None)("auth")

    def _get_from_container(container: object, key: str) -> Optional[Any]:
        if isinstance(container, dict):
            return container.get(key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except TypeError:
                    return None
        try:
            return getattr(container, key)
        except AttributeError:
            return None

    def _extract_credential(container: object, keys: Tuple[str, ...]) -> Optional[Any]:
        for key in keys:
            value = _get_from_container(container, key)
            if value is not None:
                return value
        return None

    username = None
    password = None
    if auth_section is not None:
        username = _extract_credential(auth_section, ("username", "id", "user", "name"))
        password = _extract_credential(auth_section, ("password", "pass", "pwd"))

    if username is None:
        username = get_secret_value("USERNAME") or get_secret_value("ID")
    if password is None:
        password = get_secret_value("PASSWORD") or get_secret_value("PASS")

    normalized_username = _normalize_credential(str(username)) if username is not None else None
    normalized_password = _normalize_credential(str(password)) if password is not None else None
    return normalized_username, normalized_password


def get_configured_auth_credentials() -> Tuple[str, str]:
    secret_username, secret_password = get_secret_auth_credentials()
    if secret_username and secret_password:
        return secret_username, secret_password
    return "mezamashi", "mezamashi"


def require_login() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    st.title("ログイン")

    username, password = get_configured_auth_credentials()
    if not username or not password:
        st.info("ログイン情報が未設定です。管理者に連絡してください。")
        st.stop()
        return

    with st.form("login_form", clear_on_submit=False):
        input_username = st.text_input("ID")
        input_password = st.text_input("PASS", type="password")
        submitted = st.form_submit_button("ログイン")

    if submitted:
        if input_username == username and input_password == password:
            st.session_state["authenticated"] = True
            st.success("ログインしました。")
            rerun_app()
            return
        st.error("IDまたはPASSが正しくありません。")
    st.stop()


def get_current_api_key() -> Optional[str]:
    api_key = st.session_state.get("config_api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return DEFAULT_GEMINI_API_KEY


def load_configured_api_key() -> str:
    return get_current_api_key() or ""


def decode_image_data(data: Optional[object]) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None


def _load_uploaded_file(upload) -> Tuple[Optional[bytes], Optional[str]]:
    if upload is None:
        return None, None
    try:
        data = upload.read()
    except Exception:
        return None, None
    mime_type = None
    if hasattr(upload, "type") and upload.type:
        mime_type = str(upload.type)
    else:
        name = getattr(upload, "name", "") or getattr(upload, "filename", "")
        lower = str(name).lower()
        if lower.endswith(".png"):
            mime_type = "image/png"
        elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif lower.endswith(".webp"):
            mime_type = "image/webp"
    return data if data else None, mime_type


def _load_uploaded_files(uploads: Optional[object]) -> List[Tuple[bytes, Optional[str]]]:
    if uploads is None:
        return []

    if isinstance(uploads, Sequence) and not isinstance(uploads, (bytes, bytearray, memoryview, str)):
        candidates = list(uploads)
    else:
        candidates = [uploads]

    files: List[Tuple[bytes, Optional[str]]] = []
    for upload in candidates:
        data, mime = _load_uploaded_file(upload)
        if data:
            files.append((data, mime))
    return files


def extract_parts(candidate: object) -> Sequence:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if parts is None and isinstance(candidate, dict):
        parts = candidate.get("content", {}).get("parts", [])
    return parts or []


def collect_image_bytes(response: object) -> Optional[bytes]:
    visited: set[int] = set()
    queue: List[object] = []

    if response is not None:
        queue.append(response)

    def handle_inline(container: object) -> Optional[bytes]:
        if container is None:
            return None
        data = getattr(container, "data", None)
        if data is None and isinstance(container, dict):
            data = container.get("data")
        return decode_image_data(data)

    def maybe_file_data(container: object) -> Optional[bytes]:
        if container is None:
            return None
        file_data = getattr(container, "file_data", None)
        if file_data is None and isinstance(container, dict):
            file_data = container.get("file_data")
        if file_data:
            data = getattr(file_data, "data", None)
            if data is None and isinstance(file_data, dict):
                data = file_data.get("data")
            decoded = decode_image_data(data)
            if decoded:
                return decoded
        return None

    base64_charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        if isinstance(current, bytes):
            if current:
                return current
            continue

        if isinstance(current, (bytearray, memoryview)):
            as_bytes = bytes(current)
            if as_bytes:
                return as_bytes
            continue

        if isinstance(current, str):
            candidate = current.strip()
            if len(candidate) > 80 and set(candidate) <= base64_charset:
                decoded = decode_image_data(candidate)
                if decoded:
                    return decoded
            continue

        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(current, dict):
            inline = current.get("inline_data")
            decoded = handle_inline(inline)
            if decoded:
                return decoded

            decoded = maybe_file_data(current)
            if decoded:
                return decoded

            for key, value in current.items():
                if key in {"data", "image", "blob", "bytesBase64Encoded"}:
                    decoded = decode_image_data(value)
                    if decoded:
                        return decoded
                queue.append(value)
            continue

        decoded = handle_inline(getattr(current, "inline_data", None))
        if decoded:
            return decoded

        decoded = maybe_file_data(current)
        if decoded:
            return decoded

        for attr in (
            "candidates",
            "content",
            "parts",
            "generated_content",
            "contents",
            "responses",
            "messages",
            "media",
            "image",
            "images",
        ):
            value = getattr(current, attr, None)
            if value is not None:
                queue.append(value)

        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray, memoryview)):
            queue.extend(list(current))

    return None


def detect_image_format(image_bytes: Optional[bytes]) -> Tuple[str, str]:
    default = ("png", "image/png")
    if not image_bytes:
        return default
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "").upper()
    except Exception:
        return default
    mapping = {
        "PNG": ("png", "image/png"),
        "JPEG": ("jpg", "image/jpeg"),
        "JPG": ("jpg", "image/jpeg"),
        "WEBP": ("webp", "image/webp"),
    }
    return mapping.get(fmt, default)


def collect_text_parts(response: object) -> List[str]:
    texts: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        for part in extract_parts(candidate):
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(text)
    return texts


def _get_from_container(container: object, key: str) -> Optional[Any]:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            try:
                return getter(key, None)
            except TypeError:
                return None
    try:
        return getattr(container, key)
    except AttributeError:
        return None


def sanitize_filename_component(value: str, max_length: int = 80) -> str:
    text = value or ""
    sanitized_chars: List[str] = []
    for char in text:
        if char in {"\n", "\r"}:
            sanitized_chars.append("-n-")
            continue
        if ord(char) < 32:
            continue
        if char in {'\\', '/', ':', '*', '?', '"', '<', '>', '|'}:
            continue
        if char.isspace():
            sanitized_chars.append("_")
            continue
        sanitized_chars.append(char)
    sanitized = "".join(sanitized_chars).strip("_")
    if not sanitized:
        sanitized = "prompt"
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def build_prompt_based_filename(prompt_text: str, extension: str = "png") -> str:
    prompt_component = sanitize_filename_component(prompt_text or "prompt", max_length=80)
    unique_suffix = uuid.uuid4().hex
    cleaned_ext = extension.lower().lstrip(".") or "png"
    return f"user06_{prompt_component}_{unique_suffix}.{cleaned_ext}"


def upload_image_to_gcs(
    image_bytes: bytes,
    filename_prefix: str = "gemini_image",
    object_name: Optional[str] = None,
    mime_type: str = "image/png",
    extension: str = "png",
) -> Tuple[Optional[str], Optional[str]]:
    normalized_ext = (extension or "").lower().lstrip(".") or "png"
    normalized_mime = mime_type or "image/png"
    if not is_gcs_upload_enabled():
        st.info("GCS へのアップロードは無効化されています。")
        return None, None
    if not image_bytes:
        return None, None

    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        st.warning("GCPの設定が見つからないためアップロードをスキップしました。")
        return None, None
    except Exception as exc:  # noqa: BLE001
        st.error(f"GCPの設定取得時にエラーが発生しました: {exc}")
        return None, None

    gcp_section = None
    if isinstance(secrets_obj, dict):
        gcp_section = secrets_obj.get("gcp")
    else:
        gcp_section = _get_from_container(secrets_obj, "gcp")

    if not gcp_section:
        st.warning("GCPの設定が見つからないためアップロードをスキップしました。")
        return None, None

    bucket_name = _get_from_container(gcp_section, "bucket_name")
    service_account_json = _get_from_container(gcp_section, "service_account_json")
    project_id = _get_from_container(gcp_section, "project_id")

    if not bucket_name or not service_account_json:
        st.warning("GCPの設定のうち bucket_name または service_account_json が不足しています。")
        return None, None

    service_account_info: Optional[Dict[str, Any]] = None
    if isinstance(service_account_json, (dict,)):
        service_account_info = dict(service_account_json)
    elif isinstance(service_account_json, (str, bytes)):
        raw_json = service_account_json.decode("utf-8") if isinstance(service_account_json, bytes) else service_account_json
        raw_json = raw_json.strip()
        try:
            service_account_info = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                service_account_info = json.loads(raw_json, strict=False)
            except json.JSONDecodeError as exc:
                st.error(f"service_account_json の読み込みに失敗しました: {exc}")
                return None, None
    else:
        st.error("service_account_json の形式が不明です。文字列または辞書で設定してください。")
        return None, None

    if not isinstance(service_account_info, dict):
        st.error("service_account_json の内容が辞書形式ではありません。")
        return None, None

    try:
        storage_client = storage.Client.from_service_account_info(
            service_account_info,
            project=str(project_id) if project_id else None,
        )
        bucket = storage_client.bucket(str(bucket_name))
        if object_name:
            cleaned_object_name = object_name.strip()
            if not cleaned_object_name.lower().endswith(f".{normalized_ext}"):
                cleaned_object_name = f"{cleaned_object_name}.{normalized_ext}"
            cleaned_object_name = cleaned_object_name.replace("/", "_").replace("\\", "_")
            filename = f"images/{cleaned_object_name}"
        else:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"images/{filename_prefix}_{timestamp}_{uuid.uuid4().hex}.{normalized_ext}"
        blob = bucket.blob(filename)
        blob.upload_from_file(io.BytesIO(image_bytes), content_type=normalized_mime)

        gcs_path = f"gs://{bucket.name}/{filename}"
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return gcs_path, signed_url
    except Exception as exc:  # noqa: BLE001
        st.error(f"GCSへのアップロードに失敗しました: {exc}")
        return None, None


def get_vertex_ai_settings() -> Tuple[Optional[str], Optional[str], Optional[service_account.Credentials]]:
    project_id = _normalize_credential(
        get_secret_value("VERTEX_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = _normalize_credential(
        get_secret_value("VERTEX_LOCATION")
        or os.getenv("VERTEX_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    )

    credentials: Optional[service_account.Credentials] = None
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        secrets_obj = None
    except Exception:
        secrets_obj = None

    vertex_section = None
    if isinstance(secrets_obj, dict):
        vertex_section = secrets_obj.get("vertex_ai")
    elif secrets_obj is not None:
        vertex_section = _get_from_container(secrets_obj, "vertex_ai")

    service_account_json = None
    if vertex_section:
        service_account_json = _get_from_container(vertex_section, "service_account_json")
        if not project_id:
            project_id = _normalize_credential(_get_from_container(vertex_section, "project_id") or "")
        if not location:
            location = _normalize_credential(
                _get_from_container(vertex_section, "location") or _get_from_container(vertex_section, "region") or ""
            )
    if service_account_json is None:
        service_account_json = get_secret_value("VERTEX_SERVICE_ACCOUNT_JSON") or os.getenv("VERTEX_SERVICE_ACCOUNT_JSON")

    if service_account_json:
        service_account_info: Optional[Dict[str, Any]] = None
        if isinstance(service_account_json, dict):
            service_account_info = dict(service_account_json)
        elif isinstance(service_account_json, (str, bytes)):
            raw_json = service_account_json.decode("utf-8") if isinstance(service_account_json, bytes) else service_account_json
            raw_json = raw_json.strip()
            try:
                service_account_info = json.loads(raw_json)
            except json.JSONDecodeError:
                try:
                    service_account_info = json.loads(raw_json, strict=False)
                except json.JSONDecodeError as exc:
                    st.error(f"VERTEX service_account_json の読み込みに失敗しました: {exc}")
                    service_account_info = None
        else:
            st.error("VERTEX service_account_json の形式が不明です。文字列または辞書で設定してください。")

        if isinstance(service_account_info, dict):
            try:
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
            except Exception as exc:  # noqa: BLE001
                st.error(f"VERTEX 認証情報の初期化に失敗しました: {exc}")
                credentials = None

        if not project_id and isinstance(service_account_info, dict):
            project_id = _normalize_credential(str(service_account_info.get("project_id") or ""))

    return project_id, location, credentials


def get_vertex_ai_api_key() -> Optional[str]:
    api_key = (
        get_secret_value("VERTEX_API_KEY")
        or os.getenv("VERTEX_API_KEY")
        or get_secret_value("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None

    section = None
    if isinstance(secrets_obj, dict):
        section = secrets_obj.get("vertex_ai")
    else:
        section = _get_from_container(secrets_obj, "vertex_ai")
    if section:
        candidate = _get_from_container(section, "api_key")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _vertex_image_from_bytes(image_bytes: bytes) -> VertexImage:
    if hasattr(VertexImage, "from_bytes"):
        return VertexImage.from_bytes(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        temp_path = tmp.name
    try:
        return VertexImage.load_from_file(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def upscale_image_with_vertex(
    image_bytes: bytes,
    upscale_factor: str = "x4",
    output_format: str = "png",
) -> Optional[bytes]:
    if not image_bytes:
        return None
    api_key = get_vertex_ai_api_key()
    project_id, location, credentials = get_vertex_ai_settings()
    if api_key:
        if not project_id or not location:
            st.warning("VERTEX_PROJECT_ID と VERTEX_LOCATION が必要です。")
            return None
        endpoint = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/publishers/google/models/{IMAGEN_UPSCALE_MODEL}:predict"
        )

        payload = {
            "instances": [
                {
                    "prompt": "Upscale the image",
                    "image": {
                        "bytesBase64Encoded": base64.b64encode(image_bytes).decode("utf-8")
                    },
                }
            ],
            "parameters": {
                "mode": "upscale",
                "outputOptions": {
                    "mimeType": "image/png",
                },
                "upscaleConfig": {"upscaleFactor": upscale_factor},
            },
        }
        response = requests.post(
            f"{endpoint}?key={api_key}",
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"Vertex AI API error: {response.status_code} {response.text}")
        response_json = response.json()
        extracted = collect_image_bytes(response_json)
        if extracted:
            return extracted
        raise RuntimeError("Vertex AI API のレスポンスに画像データがありません。")

    if not project_id:
        st.warning("VERTEX_PROJECT_ID が未設定です。Streamlit secrets か環境変数で設定してください。")
        return None
    if not location:
        st.warning("VERTEX_LOCATION が未設定です。Streamlit secrets か環境変数で設定してください。")
        return None

    vertexai.init(project=project_id, location=location, credentials=credentials)
    model = ImageGenerationModel.from_pretrained(IMAGEN_UPSCALE_MODEL)

    source_image = _vertex_image_from_bytes(image_bytes)
    upscaled_image = model.upscale_image(image=source_image, upscale_factor=upscale_factor)

    normalized_format = (output_format or "png").lower().lstrip(".") or "png"
    with tempfile.NamedTemporaryFile(suffix=f".{normalized_format}", delete=False) as tmp:
        temp_path = tmp.name
    try:
        upscaled_image.save(temp_path)
        with open(temp_path, "rb") as file_handle:
            return file_handle.read()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def init_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, object]] = []


def ensure_lightbox_assets() -> None:
    components.html(
        """
        <script>
        (function () {
            const parentWindow = window.parent;
            if (!parentWindow) {
                return;
            }

            try {
                delete parentWindow.__streamlitLightbox;
            } catch (err) {
                parentWindow.__streamlitLightbox = undefined;
            }
            parentWindow.__streamlitLightboxInitialized = false;
            const doc = parentWindow.document;

            if (!doc.getElementById("streamlit-lightbox-style")) {
                const style = doc.createElement("style");
                style.id = "streamlit-lightbox-style";
                style.textContent = `
                .streamlit-lightbox-thumb {
                    width: 100%;
                    display: block;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: transform 0.16s ease-in-out;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
                    margin: 0 auto 0.75rem auto;
                }
                .streamlit-lightbox-thumb:hover {
                    transform: scale(1.02);
                }
                `;
                doc.head.appendChild(style);
            }

            parentWindow.__streamlitLightbox = (function () {
                let overlay = null;
                let keyHandler = null;

                function hide() {
                    if (!overlay) {
                        return;
                    }
                    overlay.style.opacity = "0";
                    const originalOverflow = overlay.getAttribute("data-original-overflow") || "";
                    doc.body.style.overflow = originalOverflow;
                    setTimeout(function () {
                        if (overlay && overlay.parentNode) {
                            overlay.parentNode.removeChild(overlay);
                        }
                        overlay = null;
                    }, 180);
                    if (keyHandler) {
                        parentWindow.removeEventListener("keydown", keyHandler);
                        keyHandler = null;
                    }
                }

                function show(src) {
                    hide();
                    overlay = doc.createElement("div");
                    overlay.id = "streamlit-lightbox-overlay";
                    overlay.style.position = "fixed";
                    overlay.style.zIndex = "10000";
                    overlay.style.top = "0";
                    overlay.style.left = "0";
                    overlay.style.right = "0";
                    overlay.style.bottom = "0";
                    overlay.style.display = "flex";
                    overlay.style.justifyContent = "center";
                    overlay.style.alignItems = "center";
                    overlay.style.background = "rgba(0, 0, 0, 0.92)";
                    overlay.style.cursor = "zoom-out";
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.18s ease-in-out";
                    overlay.setAttribute("data-original-overflow", doc.body.style.overflow || "");
                    doc.body.style.overflow = "hidden";

                    const full = doc.createElement("img");
                    full.src = src;
                    full.alt = "Generated image fullscreen";
                    full.style.maxWidth = "100vw";
                    full.style.maxHeight = "100vh";
                    full.style.objectFit = "contain";
                    full.style.boxShadow = "0 20px 45px rgba(0, 0, 0, 0.5)";
                    full.style.borderRadius = "0";

                    overlay.appendChild(full);
                    overlay.addEventListener("click", hide);

                    keyHandler = function (event) {
                        if (event.key === "Escape") {
                            hide();
                        }
                    };
                    parentWindow.addEventListener("keydown", keyHandler);

                    doc.body.appendChild(overlay);
                    requestAnimationFrame(function () {
                        overlay.style.opacity = "1";
                    });
                }

                return { show, hide };
            })();
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_clickable_image(image_bytes: bytes, element_id: str, mime_type: str = "image/png") -> None:
    ensure_lightbox_assets()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    image_src = f"data:{mime_type};base64,{encoded}"
    image_src_json = json.dumps(image_src)
    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        img {{
            width: 100%;
            display: block;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.16s ease-in-out;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        }}
        img:hover {{
            transform: scale(1.02);
        }}
    </style>
</head>
<body>
    <img id="thumb" src="{image_src}" alt="Generated image">
    <script>
    (function() {{
        const img = document.getElementById("thumb");
        if (!img) {{
            return;
        }}

        function resizeFrame() {{
            const frame = window.frameElement;
            if (!frame) {{
                return;
            }}
            const frameWidth = frame.getBoundingClientRect().width || img.naturalWidth || img.clientWidth || 0;
            const ratio = img.naturalWidth ? (img.naturalHeight / Math.max(img.naturalWidth, 1)) : (img.clientHeight / Math.max(img.clientWidth, 1) || 1);
            const height = frameWidth ? Math.max(160, frameWidth * ratio) : (img.clientHeight || img.naturalHeight || 320);
            frame.style.height = height + "px";
        }}

        if (img.complete) {{
            resizeFrame();
        }} else {{
            img.addEventListener("load", resizeFrame);
        }}
        window.addEventListener("resize", resizeFrame);
        setTimeout(resizeFrame, 60);

        img.addEventListener("click", function() {{
            if (window.parent && window.parent.__streamlitLightbox) {{
                window.parent.__streamlitLightbox.show({image_src_json});
            }}
        }});
    }})();
    </script>
</body>
</html>
""",
        height=400,
        scrolling=False,
    )


def render_history() -> None:
    if not st.session_state.history:
        return

    st.subheader("履歴")
    for entry in st.session_state.history:
        image_bytes = entry.get("image_bytes")
        prompt_text = entry.get("prompt") or ""
        mime_type = entry.get("mime_type") or "image/png"
        extension = (entry.get("extension") or "png").lstrip(".") or "png"
        if image_bytes:
            image_id = entry.get("id")
            if not isinstance(image_id, str):
                image_id = f"img_{uuid.uuid4().hex}"
                entry["id"] = image_id
            render_clickable_image(image_bytes, image_id, mime_type=mime_type)
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        prompt_display = prompt_text.strip()
        prompt_block = (
            f'<div style="margin-top:15px; font-weight:600;">Prompt</div>'
            f'<div style="white-space:pre-wrap; background:rgba(0,0,0,0.02); '
            f'padding:10px; border-radius:8px; margin-top:6px;">'
            f'{html.escape(prompt_display) if prompt_display else "(未入力)"}'
            f"</div>"
        )
        st.markdown(prompt_block, unsafe_allow_html=True)
        meta_bits: List[str] = []
        aspect_ratio = entry.get("aspect_ratio")
        if aspect_ratio:
            meta_bits.append(f"Aspect: {aspect_ratio}")
        resolution = entry.get("resolution")
        if resolution:
            meta_bits.append(f"Resolution: {resolution}")
        model_name = entry.get("model")
        if model_name:
            meta_bits.append(f"Model: {model_name}")
        upscale_factor = entry.get("upscale_factor")
        if upscale_factor:
            meta_bits.append(f"Upscale: {upscale_factor}")
        if entry.get("reference_used"):
            meta_bits.append("Ref: yes")
        if meta_bits:
            st.caption(" / ".join(meta_bits))

        download_filename = (
            f"{sanitize_filename_component(prompt_display or 'prompt')}_{image_id}.{extension}"
        )
        st.download_button(
            "Download",
            data=image_bytes or b"",
            file_name=download_filename,
            mime=mime_type,
            key=f"download_{image_id}",
        )
        st.divider()


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🧠", layout="centered")
    init_history()
    require_login()

    st.title("FreedomStandard")

    with st.sidebar:
        st.header("Upscale (Vertex AI)")
        upscale_factor = st.radio("倍率", ("x2", "x4"), index=1, horizontal=True)
        source_choice = st.radio(
            "アップスケール元",
            ("アップロード画像", "履歴から選択", "最新生成画像"),
            index=0,
        )
        uploaded_upscale = st.file_uploader(
            "アップスケールする画像",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="upscale_upload",
        )

        selected_history_entry: Optional[Dict[str, object]] = None
        if source_choice == "履歴から選択":
            history_entries = st.session_state.get("history") or []
            if history_entries:
                labels = []
                for idx, entry in enumerate(history_entries):
                    prompt = (entry.get("prompt") or "").strip() or "(未入力)"
                    labels.append(f"{idx + 1}: {prompt[:40]}")
                selected_index = st.selectbox(
                    "履歴を選択",
                    options=list(range(len(history_entries))),
                    format_func=lambda i: labels[i],
                )
                try:
                    selected_history_entry = history_entries[int(selected_index)]
                except (IndexError, ValueError, TypeError):
                    selected_history_entry = None
            else:
                st.info("履歴がありません。")

        if st.button("選択画像をアップスケール"):
            source_bytes = None
            source_prompt = ""
            source_aspect_ratio = None
            source_reference_used = False

            if source_choice == "アップロード画像":
                data, _mime = _load_uploaded_file(uploaded_upscale)
                if not data:
                    st.warning("アップスケールする画像をアップロードしてください。")
                else:
                    source_bytes = data
                    source_prompt = "(uploaded image)"
            elif source_choice == "履歴から選択":
                if selected_history_entry and selected_history_entry.get("image_bytes"):
                    source_bytes = selected_history_entry.get("image_bytes")
                    source_prompt = selected_history_entry.get("prompt") or ""
                    source_aspect_ratio = selected_history_entry.get("aspect_ratio")
                    source_reference_used = bool(selected_history_entry.get("reference_used"))
                else:
                    st.warning("履歴から画像を選択してください。")
            else:
                last_entry = st.session_state.get("last_generated")
                if last_entry and last_entry.get("image_bytes"):
                    source_bytes = last_entry.get("image_bytes")
                    source_prompt = last_entry.get("prompt") or ""
                    source_aspect_ratio = last_entry.get("aspect_ratio")
                    source_reference_used = bool(last_entry.get("reference_used"))
                else:
                    st.info("まずは画像を生成してください。")

            if source_bytes:
                with st.spinner("アップスケール中..."):
                    try:
                        upscaled_bytes = upscale_image_with_vertex(
                            source_bytes,
                            upscale_factor=upscale_factor,
                            output_format="png",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"アップスケールに失敗しました: {exc}")
                        upscaled_bytes = None
                if upscaled_bytes:
                    image_extension, image_mime_type = detect_image_format(upscaled_bytes)
                    st.session_state.history.insert(
                        0,
                        {
                            "id": f"img_{uuid.uuid4().hex}",
                            "image_bytes": upscaled_bytes,
                            "prompt": source_prompt,
                            "model": IMAGEN_UPSCALE_MODEL,
                            "no_text": True,
                            "aspect_ratio": source_aspect_ratio,
                            "resolution": "4K",
                            "reference_used": source_reference_used,
                            "mime_type": image_mime_type,
                            "extension": image_extension,
                            "upscale_factor": upscale_factor,
                        },
                    )
                    st.success("アップスケール完了")

    api_key = load_configured_api_key()

    prompt = st.text_area("Prompt", height=150, placeholder="描いてほしい内容を入力してくださいPromptme")
    uploaded_refs = st.file_uploader(
        "Reference images (任意・複数可)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    ref_files = _load_uploaded_files(uploaded_refs)
    aspect_ratio = st.radio(
        "アスペクト比",
        IMAGE_ASPECT_RATIO_OPTIONS,
        index=IMAGE_ASPECT_RATIO_OPTIONS.index(IMAGE_ASPECT_RATIO),
        horizontal=True,
    )
    resolution_label = st.radio(
        "解像度",
        ("1K", "2K", "4K"),
        index=0,
        horizontal=True,
    )
    if st.button("Generate", type="primary"):
        if not api_key:
            st.warning("Gemini API key が設定されていません。Streamlit secrets などで設定してください。")
            st.stop()
        if not prompt.strip():
            st.warning("プロンプトを入力してください。")
            st.stop()

        client = genai.Client(api_key=api_key.strip())
        stripped_prompt = prompt.rstrip()
        prompt_components: List[str] = []
        if stripped_prompt:
            prompt_components.append(stripped_prompt)
        prompt_components.extend([DEFAULT_PROMPT_SUFFIX, NO_TEXT_TOGGLE_SUFFIX])
        prompt_for_request = "\n".join(prompt_components)

        contents_for_request: object
        if ref_files:
            # Use explicit constructors for compatibility across SDK versions.
            image_parts = [
                types.Part(inline_data=types.Blob(data=img_bytes, mime_type=mime or "image/png"))
                for img_bytes, mime in ref_files
            ]
            text_part = types.Part(text=prompt_for_request)
            contents_for_request = [types.Content(role="user", parts=[*image_parts, text_part])]
        else:
            contents_for_request = prompt_for_request

        image_config_kwargs: Dict[str, object] = {"aspect_ratio": aspect_ratio}
        image_size_key = None
        if hasattr(types.ImageConfig, "model_fields"):
            if "image_size" in getattr(types.ImageConfig, "model_fields", {}):
                image_size_key = "image_size"
        elif hasattr(types.ImageConfig, "__fields__"):
            if "image_size" in getattr(types.ImageConfig, "__fields__", {}):
                image_size_key = "image_size"
        if image_size_key:
            image_config_kwargs[image_size_key] = resolution_label

        def run_generation(include_size: bool) -> object:
            cfg_kwargs = dict(image_config_kwargs)
            if not include_size and image_size_key:
                cfg_kwargs.pop(image_size_key, None)
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents_for_request,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(**cfg_kwargs),
                ),
            )

        with st.spinner("画像を生成しています..."):
            try:
                response = run_generation(include_size=True)
            except google_exceptions.InvalidArgument as exc:
                if "Media resolution is not enabled for this model" in str(exc) and image_size_key:
                    st.info("このモデルでは解像度指定が無効でした。デフォルト解像度で再試行します。")
                    try:
                        response = run_generation(include_size=False)
                    except Exception:
                        raise exc
                else:
                    raise
            except google_exceptions.ResourceExhausted:
                st.error(
                    "Gemini API のクォータ（無料枠または請求プラン）を超えました。"
                    "しばらく待つか、Google AI Studio で利用状況と請求設定を確認してください。"
                )
                st.info("https://ai.google.dev/gemini-api/docs/rate-limits")
                st.stop()
            except google_exceptions.GoogleAPICallError as exc:
                st.error(f"API 呼び出しに失敗しました: {exc.message}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"予期しないエラーが発生しました: {exc}")
                st.stop()

        image_bytes = collect_image_bytes(response)
        if not image_bytes:
            st.error("画像データを取得できませんでした。")
            st.stop()

        image_extension, image_mime_type = detect_image_format(image_bytes)
        user_prompt = prompt.strip()
        object_name = build_prompt_based_filename(user_prompt, extension=image_extension)
        upload_image_to_gcs(
            image_bytes,
            object_name=object_name,
            mime_type=image_mime_type,
            extension=image_extension,
        )

        st.session_state.history.insert(
            0,
            {
                "id": f"img_{uuid.uuid4().hex}",
                "image_bytes": image_bytes,
                "prompt": user_prompt,
                "model": MODEL_NAME,
                "no_text": True,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution_label,
                "reference_used": bool(ref_files),
                "mime_type": image_mime_type,
                "extension": image_extension,
            },
        )
        st.session_state["last_generated"] = {
            "image_bytes": image_bytes,
            "prompt": user_prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution_label,
            "reference_used": bool(ref_files),
        }
        st.success("生成完了")

    render_history()


if __name__ == "__main__":
    main()
