# Fruit & Vegetable Classifier

Streamlit app that classifies 51 fruits/vegetables using a pretrained ResNet-50 model.

## Model
- Pretrained weights: `fruits_vegetables_51.pth`
- Source: Kaggle model by sunnyagarwal427444 — https://www.kaggle.com/models/sunnyagarwal427444/food-ingredient-classification-model
- Architecture: ResNet-50 with a 51-class head.

## Requirements
- Python (runtime pinned to 3.13 for Streamlit Cloud)
- `requirements.txt` installs: torch 2.6.0+cpu, torchvision 0.21.0+cpu, Pillow, Streamlit.

## Local setup
```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# or: source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Usage
- Upload image: select a jpg/png → Predict.
- Camera: use “Take a picture” → Predict.
- Device auto-selects CUDA if available, otherwise CPU.

## Deploy to Streamlit Cloud
- Ensure `requirements.txt` and `runtime.txt` are in repo root.
- App entrypoint: `streamlit_app.py`.
- Model file `fruits_vegetables_51.pth` must be in the repo root (fits under 100 MB limit).
