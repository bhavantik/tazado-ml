# tazado-ml
Machine Learning Model and development pipeline

Perfect—let’s do this like a **clean, reproducible setup** you can follow line-by-line.

We’ll cover:

✅ Initialize project with **uv**
✅ Create virtual environment
✅ Install dependencies
✅ Create full folder structure
✅ Add basic starter files

---

# 🚀 STEP 1: Create Project Folder

```bash
mkdir tazado-ml
cd tazado-ml
```

---

# ⚙️ STEP 2: Initialize Project with UV

Install uv (if not already):

```bash
pip install uv
```

Create virtual environment:

```bash
uv venv
```

Activate it:

### Mac/Linux:

```bash
source .venv/bin/activate
```

### Windows:

```bash
.venv\Scripts\activate
```

---

# 📦 STEP 3: Initialize Python Project (pyproject.toml)

```bash
uv init
```

👉 This creates:

```bash
pyproject.toml
```

---

# 📦 STEP 4: Create requirements.txt

```bash
touch requirements.txt
```

Add this content:

```txt
tensorflow
tensorflow-lite
numpy
pandas
opencv-python
matplotlib
scikit-learn
pyyaml
tqdm
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

# 🏗️ STEP 5: Create Folder Structure

Run this (Mac/Linux):

```bash
mkdir -p data/raw data/processed data/splits \
models/mango_binary models/mango_variety \
configs \
src/data src/models src/training src/inference src/utils \
notebooks tests
```

---

### 🪟 Windows (PowerShell):

```powershell
mkdir data, data\raw, data\processed, data\splits
mkdir models, models\mango_binary, models\mango_variety
mkdir configs
mkdir src, src\data, src\models, src\training, src\inference, src\utils
mkdir notebooks, tests
```

---

# 📄 STEP 6: Create Base Files

## 1. README

```bash
touch README.md
```

---

## 2. Model Registry

```bash
touch models/registry.json
```

Add:

```json
{
  "mango_binary": {
    "version": "v1",
    "path": "models/mango_binary/v1/model.tflite"
  },
  "mango_variety": {
    "version": "v1",
    "path": "models/mango_variety/v1/model.tflite"
  }
}
```

---

## 3. Config Files

```bash
touch configs/base.yaml
touch configs/mango_binary.yaml
touch configs/mango_variety.yaml
```

Example (`mango_binary.yaml`):

```yaml
model: mobilenetv3
epochs: 10
batch_size: 32
learning_rate: 0.001
image_size: 224
```

---

# 🧠 STEP 7: Create Starter Python Files

## Data Loader

```bash
touch src/data/loader.py
```

---

## Model File

```bash
touch src/models/mobilenet.py
```

---

## Training Script

```bash
touch src/training/train.py
```

---

## Inference Script

```bash
touch src/inference/predict.py
```

---

## Utils

```bash
touch src/utils/config.py
touch src/utils/logger.py
```

---

# 📁 STEP 8: Dataset Structure

Manually create:

```bash
data/raw/mango/kesar/
data/raw/mango/alphonso/
data/raw/mango/langda/
data/raw/not_mango/
```

---

# 🧪 STEP 9: Test Setup

Run:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

👉 If version prints → setup is correct ✅

---

# 🔥 STEP 10: First Commit (Git)

```bash
git init
git add .
git commit -m "Initial ML project setup with uv and structure"
```

---

# ⚠️ IMPORTANT (Don’t Skip)

Create `.gitignore`:

```bash
touch .gitignore
```

Add:

```txt
.venv/
data/
models/
*.pyc
__pycache__/
```

---

# 🧠 What You Achieved

You now have:

✅ Scalable ML repo
✅ Multi-model ready
✅ Clean structure for future
✅ No refactor needed later

---

# 🚀 Next Step

Say:

👉 **“give me code for mango_binary training using MobileNet”**

I’ll give you:

* full working training code
* dataset loader
* model definition
* training loop
* TFLite export

Let’s build your first model 🔥
