---
license: mit
---
# VoxSentinel-Emotion-Base (it has not yet reach my standard but still acceptable)🎭

### HuggingFace Repo: https://huggingface.co/JesseHuang922/VoxSentinel-Emotion-Base

**VoxSentinel-Emotion-Base** is a robust Speech Emotion Recognition (SER) model based on the **Wav2Vec2** architecture. It is designed to classify human speech into seven core emotional states with high precision, balancing performance across high-quality studio recordings and noisy, real-world conversational data.

## 🚀 Key Features
- **Backbone**: Wav2Vec2-Base (Fine-tuned).
- **Pooling Mechanism**: Multi-Scale Attentive Pooling (captures both weighted Mean and Standard Deviation of speech features).
- **Emotions Supported**: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`.
- **Target Sampling Rate**: 16kHz.

## 📊 Performance Summary
The model achieves strong generalization across multiple benchmark datasets:
- **CREMA-D / RAVDESS / TESS**: ~75% - 100% Accuracy (Clean Environment).
- **MELD**: Optimized for robustness against background noise and varied recording conditions.

---

## 🛠️ How to Use

### 1. Installation
Ensure you have the necessary libraries installed:
```bash
pip install torch transformers librosa
```

### 2. Inference Script
You can pull the model directly from the Hub and run inference using the following code:

```python
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, AutoProcessor

# 1. Define the Architecture (Required for loading custom layers)
class MultiScaleAttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        mu = torch.sum(attn_weights * x, dim=1)
        delta = x - mu.unsqueeze(1)
        var = torch.sum(attn_weights * (delta ** 2), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mu, std], dim=-1)

class VoxSentinelForEmotion(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pooling = MultiScaleAttentivePooling(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)
        )
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state)
        logits = self.classifier(pooled_output)
        return logits

# 2. Load Model and Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_id = "JesseHuang922/VoxSentinel-Emotion-Base"

model = VoxSentinelForEmotion.from_pretrained(repo_id).to(device)
processor = AutoProcessor.from_pretrained(repo_id)
model.eval()

# 3. Predict Function
def predict(audio_path):
    # Load and resample to 16kHz
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    
    with torch.no_grad():
        logits = model(inputs)
        pred_idx = torch.argmax(logits, dim=1).item()
    
    return model.config.id2label[pred_idx]

# Example Usage
# emotion = predict("your_audio_file.wav")
# print(f"Detected Emotion: {emotion}")
```

### 📂 Model Structure

config.json: Model configuration and label mapping.

model.safetensors: Optimized model weights.

preprocessor_config.json: Audio preprocessing parameters.