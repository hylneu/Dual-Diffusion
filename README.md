# Dual-Diffusion  
![Screen](image/Figure.png)

> **Subject–Driven · Personalized · Text-to-Image Generation**  
> Built on top of *Stable Diffusion 2-1* and *DreamBench.*

---

## ✨ Highlights

- **🔄 Dual Training Pipeline** – Combines *Subject Inversion* and *Prompt Editing* for richer personalization.
- **⚖️ Lightweight & Fast** – Fine-tune in **< 30 min** on a single 16 GB GPU.
- **📊 SOTA Quality** – Outperforms DreamBooth, LoRA, IP-Adapter, DisenBooth & more on DreamBench.
- 
## 🛠️ Requirements

-  **diffusers** `>=0.23.1`
-  **open_clip_torch**
-  **torchvision**
-  **Hardware**: at least **1 × 16 GB** NVIDIA GPU

## 🏋️ Training

<pre> ```bash bash train.sh ``` </pre>

## 📚 Datasets
Download the [DreamBench Dataset] (https://github.com/google/dreambooth) and extract them to dataset/\
Download Pretrained Model Stable Diffusion 2-1.
