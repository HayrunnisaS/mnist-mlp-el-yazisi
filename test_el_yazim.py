import torch
import torch.nn as nn
import cv2
import numpy as np

# Model yapisi
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


# Egitilmis modeli yukle
model = MLP()
model.load_state_dict(
    torch.load("model_deney1.pth", map_location="cpu")
)
model.eval()


# Fotograf dosyasini oku
image = cv2.imread("okul_numaram.jpg")

if image is None:
    image = cv2.imread("okul_numaram.png")

if image is None:
    raise FileNotFoundError(
        "okul_numaram.jpg veya okul_numaram.png bulunamadi!"
    )


# Gri tonlama ve gurultu azaltma
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Rakamlarin arka plandan ayrilmasi
if np.mean(gray) > 127:
    threshold_type = cv2.THRESH_BINARY_INV
else:
    threshold_type = cv2.THRESH_BINARY

_, binary = cv2.threshold(
    gray,
    0,
    255,
    threshold_type + cv2.THRESH_OTSU
)

# Mevcut threshold kodunuz:
# _, binary = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)

# --- ALTINA BU KISMI EKLEYİN ---
# Çizgileri kalınlaştırmak için Dilation (Genişletme) işlemi
kernel = np.ones((2, 2), np.uint8)
binary = cv2.dilate(binary, kernel, iterations=1)
# ------------------------------

# Rakam parcalarini bul
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

height, width = gray.shape
digits = []

min_height = max(12, int(height * 0.025))
min_area = max(25, int(height * width * 0.00001))

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    if h >= min_height and w >= 3 and area >= min_area:
        digits.append((x, y, w, h))


# --- YENI SIRALAMA MANTIGI (Satir satir okuma) ---
if digits:
    # 1. Once Y eksenine (yukaridan asagiya) gore sirala
    digits.sort(key=lambda item: item[1])

    rows = []
    current_row = [digits[0]]

    # 2. Y ekseninde birbirine yakin olanlari (ayni hizada olanlari) ayni satir olarak grupla
    for box in digits[1:]:
        # Eger y eksenindeki fark 150 pikselden kucukse ayni satirdadir (gorselin boyutuna gore ayarlanabilir)
        if abs(box[1] - current_row[-1][1]) < 150:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    # 3. Her satirin icindeki rakamlari X eksenine (soldan saga) gore kendi icinde sirala
    sorted_digits = []
    for row in rows:
        sorted_digits.extend(sorted(row, key=lambda item: item[0]))
    
    digits = sorted_digits
# -------------------------------------------------


# Her rakami model icin hazirla ve tahmin et
predictions = []

print("\nOKUL NUMARASI TAHMIN SONUCLARI\n")

for index, (x, y, w, h) in enumerate(digits, start=1):
    crop = binary[y:y + h, x:x + w]

    # Rakam oranini koruyarak 20x20 alana sigdir
    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    crop = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # MNIST boyutunda siyah tuval
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Yazim hatasi duzeltildi: (28 - new_w)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = crop

    tensor = torch.tensor(
        canvas,
        dtype=torch.float32
    ).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()

    predictions.append(str(prediction))

    print(
        f"{index}. parca: Tahmin = {prediction} "
        f"(x={x}, y={y}, w={w}, h={h})"
    )


# Sonuclari goster
print("\nBulunan parca sayisi:", len(digits))
print("Tahmin edilen numara:", "".join(predictions))
print("\nTest tamamlandi.")