# 🎯 Tazado ML - Setup & Quick Start Guide

## ✅ Models Initialized Successfully!

You now have two pretrained models ready:

### **Stage 1: Mango Detection** (`fruit_detection.h5`)
- Binary classification: Mango vs Not Mango
- MobileNetV3Small backbone with ImageNet weights
- Size: ~4.8 MB
- Status: ✅ Ready for training

### **Stage 2: Variety Classification** (`fruit_variety.h5`)
- Multi-class classification: Alphonso, Kesar, Langda, Other
- MobileNetV3Small backbone with ImageNet weights
- Size: ~4.8 MB
- Status: ✅ Ready for training

---

## 🚀 Next Steps

### 1️⃣ **Organize Training Data**

Create the following structure:

```
data/raw/mango/
├── alphonso/          # Add 100+ mango images
├── kesar/             # Add 100+ mango images
├── langda/            # Add 100+ mango images
└── other/             # Add 50+ mango images
```

### 2️⃣ **Train the Models**

```bash
python main.py train --epochs 50 --batch_size 32
```

This will:
- Load images from `data/raw/mango/`
- Split into train/val/test sets
- Apply data augmentation
- Fine-tune both models
- Save checkpoints to `models/checkpoints/`
- Save final models to `models/`

### 3️⃣ **Test Inference**

Once training is done, test on a single image:

```bash
python main.py predict --image path/to/test_image.jpg
```

Output:
```
Mango detected! Variety: alphonso (0.95 confidence)
```

---

## 📋 Available Commands

```bash
# Initialize models (already done!)
python main.py init_models

# Train on your data
python main.py train --epochs 50 --batch_size 32

# Predict on single image
python main.py predict --image path/to/image.jpg

# Predict on batch of images
python main.py predict_batch --dir path/to/images/

# Test models
python main.py test_models
```

---

## 🏗️ Project Structure

```
tazado-ml/
├── models/
│   ├── fruit_detection.h5      ✅ Stage 1 model
│   ├── fruit_variety.h5        ✅ Stage 2 model
│   ├── registry.json           ✅ Model metadata
│   └── checkpoints/            (Created during training)
├── data/
│   ├── raw/mango/              (Add your training images here)
│   │   ├── alphonso/
│   │   ├── kesar/
│   │   ├── langda/
│   │   └── other/
│   ├── processed/              (Auto-created during training)
│   └── splits/                 (Auto-created during training)
├── src/
│   ├── data/loader.py          Data loading & augmentation
│   ├── models/mobilenet.py     Model architecture
│   ├── training/
│   │   ├── train.py            Training pipeline
│   │   └── evaluate.py         Model evaluation
│   ├── inference/predict.py    Inference interface
│   └── utils/
│       ├── config.py           Configuration
│       └── logger.py           Logging
├── init_models.py              ✅ Model initialization
├── test_models.py              Model testing
├── main.py                     CLI interface
├── config.yaml                 Training config
└── requirements.txt            Dependencies
```

---

## 📊 Model Architecture

### Both Models Use:
- **Backbone**: MobileNetV3Small
- **Pretrained Weights**: ImageNet
- **Input Size**: 224×224×3 pixels
- **Framework**: TensorFlow/Keras

### Stage 1 (Binary)
```
Input (224×224×3)
  ↓
MobileNetV3Small backbone
  ↓
Global Average Pooling
  ↓
Dense(256) + Dropout(0.3)
  ↓
Dense(128) + Dropout(0.2)
  ↓
Output (Sigmoid) → [0, 1]
```

### Stage 2 (Multi-class)
```
Input (224×224×3)
  ↓
MobileNetV3Small backbone
  ↓
Global Average Pooling
  ↓
Dense(256) + Dropout(0.3)
  ↓
Dense(128) + Dropout(0.2)
  ↓
Output (Softmax) → [4 classes]
```

---

## 💡 Training Tips

| Issue | Solution |
|-------|----------|
| **Slow training** | Reduce `--batch_size` or use smaller epochs |
| **Out of memory** | Set `export CUDA_VISIBLE_DEVICES=-1` to use CPU |
| **Low accuracy** | Collect more diverse training images |
| **Overfitting** | Increase data augmentation or dropout |
| **Fast inference needed** | Export to TFLite for mobile |

---

## 📊 Expected Performance

With 300+ images per variety and proper augmentation:
- **Stage 1 Accuracy**: 95%+
- **Stage 2 Accuracy**: 90%+
- **Training Time**: 30-60 minutes (depends on hardware)

---

## 🔄 Inference Pipeline

```
Input Image
    ↓
Preprocess (224×224, normalize)
    ↓
Stage 1: Mango Detection
    ├─ If mango_confidence < 0.5
    │  └─ Return: "Not a mango"
    └─ If mango_confidence >= 0.5
       ↓
       Stage 2: Variety Classification
           ↓
           Return: {variety, confidence}
```

---

## 📈 Model Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  input_size: 224
  dropout_rate: 0.3

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  validation_split: 0.2

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

---

## 🚀 Ready to Train!

Your models are initialized and ready. Next:

1. Add training images to `data/raw/mango/`
2. Run: `python main.py train`
3. Use trained models: `python main.py predict --image image.jpg`

Happy training! 🎉
