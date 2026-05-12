# Deep-AE Deployment Notes

This repository has been strictly structured to support multiple deployment architectures. Below are the recommended deployment strategies depending on your target environment.

## 1. Streamlit Community Cloud (Recommended for Demo/Portfolio)

Streamlit Community Cloud is the easiest way to host the Deep-AE dashboard publicly for free.

**Requirements:**
- A GitHub account
- The `requirements.txt` file (already optimized in root)
- The `app.py` file in the root directory.

**Steps:**
1. Push this repository to a public or private GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repository, branch (e.g., `main`), and set the main file path to `app.py`.
5. Click **Deploy**.

**Notes:** 
Streamlit Cloud allocates ~1GB of RAM. The `deepae_cnn.pth` model is lightweight and in-memory audio processing (via `librosa` and `io.BytesIO`) ensures we do not exceed this limit.

---

## 2. Docker Deployment (Containerized Production)

For production environments (AWS, GCP, Azure), containerizing the application ensures consistent behavior.

**Example `Dockerfile`:**
```dockerfile
FROM python:3.10-slim

# Install libsndfile for audio processing
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
*Note: This Dockerfile is provided as a template for future scope.*

---

## 3. Edge Deployment (Raspberry Pi / IoT)

To deploy the inference pipeline onto a constrained edge device (e.g., Raspberry Pi 4 attached to structural sensors), it is highly recommended to convert the PyTorch model to ONNX format.

**Why ONNX?**
PyTorch requires a large installation footprint. ONNX Runtime is extremely lightweight and faster for CPU-bound edge devices.

**Basic Conversion Script:**
```python
import torch
from models.model import DeepAE_CNN

model = DeepAE_CNN(num_classes=2)
model.load_state_dict(torch.load("models/deepae_cnn.pth", map_location="cpu")["model_state_dict"])
model.eval()

# Create dummy input tensor matching the required shape
dummy_input = torch.randn(1, 1, 128, 128)

torch.onnx.export(
    model, 
    dummy_input, 
    "models/deepae_cnn.onnx", 
    export_params=True, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output']
)
```

Once converted, you can run `predict.py` using `onnxruntime` instead of `torch`, significantly reducing RAM usage.
