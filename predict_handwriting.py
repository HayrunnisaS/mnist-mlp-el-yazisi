
import cv2
import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

device = torch.device("cpu")
model = MLP().to(device)
model.load_state_dict(torch.load(
    "model_deney1.pth", map_location=device
))
model.eval()

image = cv2.imread("el_yazim.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    image = cv2.imread("el_yazim.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        "el_yazim.jpg veya el_yazim.png bulunamadi!"
    )

_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

digits = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w * h < 80:
        continue

    digit = binary[y:y+h, x:x+w]

    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit = cv2.resize(
        digit, (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    digits.append((x, canvas))

digits.sort(key=lambda item: item[0])

if not digits:
    raise ValueError("Fotoğrafta rakam bulunamadi.")

print("Bulunan rakam sayisi:", len(digits))

with torch.no_grad():
    for i, (x, digit) in enumerate(digits):
        tensor = torch.tensor(
            digit.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)

        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item() * 100

        print(
            f"Rakam {i+1}: Tahmin = {prediction}, "
            f"Guven = %{confidence:.2f}"
        )
