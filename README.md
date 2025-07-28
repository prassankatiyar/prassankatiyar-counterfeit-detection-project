# Counterfeit Sneaker Detection

An AI model that detects fake sneakers using computer vision. This project covers the full workflow from training and interpretability (using Grad-CAM) to optimization for mobile devices (using quantization).

---

## 🔍 Model

A **PyTorch-based** computer vision model fine-tuned for high-accuracy classification.

---

## 🧠 Interpretability (XAI)

Implements **Grad-CAM** to generate heatmaps, visually explaining the model's predictions by highlighting key features like **logos** and **stitching**.

---

## ⚙️ Edge Optimization

The model is **quantized** from **FP32** to **INT8**, significantly reducing its size and increasing inference speed with minimal impact on accuracy.

---

## 🖼️ Grad-CAM Result

The heatmap shows the model focusing on the **logo** to make its prediction.