from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from loadmodel import CLASS_NAMES, TRANSFORM, load_model


DEFAULT_WEIGHTS = Path("fruits_vegetables_51.pth")


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@st.cache_resource(show_spinner=False)
def get_model(weights_path: Path) -> tuple[torch.nn.Module, torch.device]:
    device = get_device()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model = load_model(weights_path, device)
    return model, device


def run_inference(image: Image.Image, model: torch.nn.Module, device: torch.device) -> str:
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        predicted_class = outputs.argmax(dim=1).item()
    return CLASS_NAMES[predicted_class]


def render_upload() -> None:
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a fruit/vegetable photo to get a prediction.")
        return

    try:
        image = Image.open(uploaded)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Could not read image: {exc}")
        return

    st.image(image, caption="Input image", use_column_width=True)

    if st.button("Predict", type="primary", key="predict_upload"):
        with st.spinner("Loading model and running inference..."):
            try:
                model, device = get_model(DEFAULT_WEIGHTS)
                prediction = run_inference(image, model, device)
                st.success(f"Predicted class: **{prediction}**")
                st.caption(f"Device: {device}")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(str(exc))


def render_camera() -> None:
    photo = st.camera_input("Take a picture")
    if photo is None:
        st.info("Click 'Take photo' and show the fruit/vegetable.")
        return

    try:
        image = Image.open(photo)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Could not read camera image: {exc}")
        return

    st.image(image, caption="Captured photo", use_column_width=True)

    if st.button("Predict", type="primary", key="predict_camera"):
        with st.spinner("Running prediction..."):
            try:
                model, device = get_model(DEFAULT_WEIGHTS)
                prediction = run_inference(image, model, device)
                st.success(f"Predicted class: **{prediction}**")
                st.caption(f"Device: {device}")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="Fruit & Vegetable Classifier", page_icon=":leafy_green:", layout="centered")
    st.title("Fruit & Vegetable Classifier")
    st.caption("51-class ResNet50 model (fruits_vegetables_51.pth)")

    upload_tab, camera_tab = st.tabs(["Upload image", "Camera (take photo)"])

    with upload_tab:
        render_upload()
    with camera_tab:
        render_camera()


if __name__ == "__main__":
    main()
