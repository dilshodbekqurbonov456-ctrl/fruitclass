import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# Class labels in order of training indices
CLASS_NAMES = [
    'Amaranth',
    'Apple',
    'Banana',
    'Beetroot',
    'Bell pepper',
    'Bitter Gourd',
    'Blueberry',
    'Bottle Gourd',
    'Broccoli',
    'Cabbage',
    'Cantaloupe',
    'Capsicum',
    'Carrot',
    'Cauliflower',
    'Chilli pepper',
    'Coconut',
    'Corn',
    'Cucumber',
    'Dragon_fruit',
    'Eggplant',
    'Fig',
    'Garlic',
    'Ginger',
    'Grapes',
    'Jalepeno',
    'Kiwi',
    'Lemon',
    'Mango',
    'Okra',
    'Onion',
    'Orange',
    'Paprika',
    'Pear',
    'Peas',
    'Pineapple',
    'Pomegranate',
    'Potato',
    'Pumpkin',
    'Raddish',
    'Raspberry',
    'Ridge Gourd',
    'Soy beans',
    'Spinach',
    'Spiny Gourd',
    'Sponge Gourd',
    'Strawberry',
    'Sweetcorn',
    'Sweetpotato',
    'Tomato',
    'Turnip',
    'Watermelon',
]

# ImageNet-style preprocessing used during training
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load ResNet50 with custom head and weights."""
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image_path: Path, model: torch.nn.Module, device: torch.device) -> str:
    """Run a single image prediction and return class name."""
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        predicted_class = outputs.argmax(dim=1).item()
    return CLASS_NAMES[predicted_class]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fruit/vegetable class for a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to input image (jpg/png).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("fruits_vegetables_51.pth"),
        help="Path to model weights file.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run on; default auto selects cuda if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = load_model(args.weights, device)
    prediction = predict(args.image, model, device)
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
