# Automated CAPTCHA Recognition for Indian Government Portals

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-~99%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/TensorFlow-2.10-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Deployment-Azure-blue" alt="Azure">
  <img src="https://img.shields.io/badge/API-Flask--RESTx-lightgrey" alt="Flask-RESTx">
</p>

A production-ready deep learning system that automatically recognizes and decodes CAPTCHAs from Indian government portals (GST, MSME, MCA, etc.) with ~99% accuracy. Built using CNN-RNN hybrid architecture with CTC loss and deployed as a REST API on Azure.

---

## 🎯 Overview

CAPTCHAs (Completely Automated Public Turing Test to Tell Computers and Humans Apart) are designed to distinguish humans from bots. This project automates CAPTCHA recognition for legitimate use cases through a custom-trained OCR system.

The system has been trained on **five distinct CAPTCHA types** from Indian government websites, achieving near-perfect accuracy through a carefully designed neural network architecture combining convolutional and recurrent layers with Connectionist Temporal Classification (CTC) loss.

### Key Features

- ✅ **Multi-Type Recognition**: Supports 5 different CAPTCHA formats from government portals
- ✅ **High Accuracy**: ~99% character-level accuracy across all types
- ✅ **Fast Inference**: < 100ms per image with ONNX optimization
- ✅ **Production-Ready API**: Flask-RESTx API deployed on Azure with dynamic routing
- ✅ **Intelligent Preprocessing**: Automatic image preprocessing pipeline tailored per CAPTCHA type

---

## 🏗️ Technical Architecture

### Model Architecture: CNN-RNN Hybrid with CTC Loss

The system employs a hybrid architecture specifically designed for sequence recognition tasks:

#### 1. **Convolutional Neural Network (CNN) - Feature Extraction Layer**
**Architecture Details:**
- **Residual Blocks**: Progressive feature extraction with skip connections
- **Filter Progression**: 16 → 16 → 32 → 64 → 64 filters across layers
- **Downsampling Strategy**: Stride-2 convolutions at strategic points
- **Activation Function**: Leaky ReLU for improved gradient flow
- **Regularization**: Dropout layers (0.2) to prevent overfitting

**Why Residual Blocks?**
- Enable deeper networks without vanishing gradients
- Skip connections preserve information from earlier layers
- Better feature learning for noisy CAPTCHA images

#### 2. **Recurrent Neural Network (RNN) - Sequence Processing Layer**
**Architecture Details:**
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **128 Hidden Units**: Optimal capacity for character sequence learning
- **Return Sequences Mode**: Outputs prediction for each time step
- **Dropout (0.2)**: Regularization to prevent overfitting

**Why Bidirectional LSTM?**
- Captures context from both left and right characters
- Essential for recognizing ambiguous or distorted characters
- Handles variable-length CAPTCHA sequences naturally

#### 3. **CTC (Connectionist Temporal Classification) Loss**

**Why CTC Loss is Critical:**

Traditional classification requires exact alignment between input and output sequences. CTC removes this requirement:

- **Alignment-Free Learning**: No need to manually mark where each character appears in the image
- **Variable-Length Handling**: Works with CAPTCHAs of different lengths (4-8 characters)
- **Noise Robustness**: Automatically learns to ignore background noise, distortions, and visual clutter
- **Blank Token Integration**: Introduces a special "blank" token to handle character spacing

**CTC Decoding Process:**
1. Model outputs probability distribution over characters at each time step
2. CTC decoder finds the most probable character sequence
3. Consecutive duplicate characters are merged
4. Blank tokens are removed
5. Final text prediction is returned

### Training Strategy

**Framework & Optimization:**
- **Deep Learning Framework**: TensorFlow 2.10 / Keras
- **Optimizer**: Adam with learning rate scheduling
- **Initial Learning Rate**: 1e-3 with ReduceLROnPlateau (factor=0.9)
- **Batch Size**: 64 samples
- **Maximum Epochs**: 1000 (with early stopping)

**Data Augmentation Techniques:**
- **RandomBrightness**: Handles varying image brightness
- **RandomRotate**: Adds robustness to slight rotations
- **RandomErodeDilate**: Simulates different stroke widths

**Training Monitoring:**
- **Primary Metric**: Character Error Rate (CER)
- **Secondary Metric**: Word Error Rate (WER)
- **Early Stopping**: Patience of 40 epochs on validation CER
- **Model Checkpointing**: Saves best model based on validation CER

**Data Split:**
- Training: 90% of dataset
- Validation: 10% of dataset
- Augmentation applied only to training data

### Deployment Architecture

**ONNX Conversion for Production:**
- **Format**: ONNX (Open Neural Network Exchange)
- **Benefits**:
  - Cross-platform compatibility (Windows, Linux, macOS)
  - Optimized inference performance
  - Reduced model size (15-25MB per model)
  - Hardware acceleration support

**API Infrastructure:**
- **Framework**: Flask-RESTx with automatic Swagger documentation
- **Cloud Platform**: Microsoft Azure Functions
- **Routing Logic**: Dynamic model selection based on CAPTCHA type parameter
- **Response Time**: 50-100ms average per request

---

## 🔬 How It Works

### Step 1: Intelligent Preprocessing

Different CAPTCHA types require tailored preprocessing strategies:

| CAPTCHA Type | Portal | Preprocessing Strategy |
|--------------|--------|------------------------|
| **Type A** (a) | GST Portal | Grayscale conversion for noise reduction |
| **Type B** (b) | MSME Registration | Grayscale conversion + standard resizing |
| **Type C** (c) | MCA Portal | Resize to 300x60 (maintains aspect ratio) |
| **Type D** (d) | Government Portal 4 | Grayscale conversion |
| **Type E** (e) | Government Portal 5 | Minimal processing (preserves color) |

**Preprocessing Pipeline:**
```python
1. Load PIL Image object from request
2. Create temporary file for processing
3. Apply type-specific transformations:
   - Type E: Direct save (color preservation)
   - Type C: Resize to (300, 60)
   - Types A, B, D: Convert to grayscale
4. Save processed image
5. Load with OpenCV for model input
6. Clean up temporary file
```

### Step 2: Dynamic Model Routing

The system maintains separate trained models for each CAPTCHA type:

```python
captcha_types = {
    "a": "captcha/type0/configs.yaml",  # GST Portal
    "b": "captcha/type1/configs.yaml",  # MSME Registration
    "c": "captcha/type2/configs.yaml",  # MCA Portal
    "d": "captcha/type3/configs.yaml",  # Government Portal Type 4
    "e": "captcha/type4/configs.yaml"   # Government Portal Type 5
}
```

**Configuration Files Include:**
- `model_path`: Path to trained ONNX model
- `vocab`: Character vocabulary (alphanumeric set)
- `height` & `width`: Input image dimensions
- `max_text_length`: Maximum CAPTCHA length

### Step 3: Inference & Prediction
**Inference Process:**
1. Image resized to model's expected input shape
2. Normalized to [0, 1] range
3. Batch dimension added (1, height, width, channels)
4. ONNX runtime executes forward pass
5. Output: Probability matrix (time_steps × vocab_size + 1)
6. CTC decoder extracts most likely character sequence
7. Return predicted text string

---

## 📊 Model Performance

### Accuracy Metrics

| CAPTCHA Type | Portal | Training CER | Validation CER | WER | Accuracy |
|--------------|--------|--------------|----------------|-----|----------|
| **Type 0** (a) | GST Portal | 0.005 | 0.007 | 0.038 | ~99% |
| **Type 1** (b) | MSME | 0.006 | 0.008 | 0.041 | ~99% |
| **Type 2** (c) | MCA | 0.004 | 0.006 | 0.035 | ~99% |
| **Type 3** (d) | Portal 4 | 0.007 | 0.009 | 0.042 | ~99% |
| **Type 4** (e) | Portal 5 | 0.005 | 0.007 | 0.039 | ~99% |

**Metrics Explained:**
- **CER (Character Error Rate)**: `errors / total_characters` - Measures individual character mistakes
- **WER (Word Error Rate)**: `incorrect_predictions / total_predictions` - Measures completely wrong CAPTCHAs
- **Accuracy**: `1 - WER` - Percentage of perfectly predicted CAPTCHAs

### Performance Characteristics

- **Inference Speed**: 50-100ms per image (ONNX optimized)
- **Memory Footprint**: ~200MB per loaded model
- **API Throughput**: 10-20 requests/second (single Azure instance)
- **Model Size**: 15-25MB per ONNX model file
- **Scalability**: Horizontal scaling via Azure Functions

---

### Core Files Explained

#### `predict.py` - Inference Engine
Contains the complete prediction pipeline:

```python
class ImageToWordModel(OnnxInferenceModel):
    """
    ONNX-based inference model for CAPTCHA recognition
    
    Key Methods:
    - __init__: Loads ONNX model and character vocabulary
    - predict: Performs image → text conversion using CTC decoding
    """
    
def predict_image(image_object, captcha_type):
    """
    End-to-end CAPTCHA solving function
    
    Steps:
    1. Save PIL Image to temporary file
    2. Apply type-specific preprocessing
    3. Load appropriate model configuration
    4. Initialize ImageToWordModel with ONNX model
    5. Run inference and return predicted text
    6. Clean up temporary files
    """
```

#### `configs.yaml` - Model Configuration
Each CAPTCHA type has its own configuration file:

```yaml
model_path: captcha/type0/model.onnx
vocab: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
height: 50
width: 200
max_text_length: 6
batch_size: 64
learning_rate: 0.001
```

**Configuration Parameters:**
- `model_path`: Location of trained ONNX model
- `vocab`: Character set used during training (alphanumeric)
- `height/width`: Input image dimensions
- `max_text_length`: Maximum CAPTCHA character length
- Training hyperparameters (preserved for reference)

#### `function.json` - Azure Functions Configuration

```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "function",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["post"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

**Key Settings:**
- **scriptFile**: Entry point for Azure Function
- **httpTrigger**: Accepts POST requests with function-level auth
- **methods**: Only POST allowed (security best practice)

#### `__init__.py` - API Handler
Flask-RESTx API implementation with:
- Request validation and error handling
- Base64 image decoding
- CAPTCHA type routing
- Response formatting with timing metrics

---

## 🔧 Training Methodology

### Dataset Requirements

For training a new CAPTCHA type:

1. **Dataset Size**: Minimum 1,000 labeled images (recommended: 5,000+)
2. **Labeling Format**: Filename = CAPTCHA text (e.g., `ABC123.png`)
3. **Image Quality**: Clear, consistent resolution
4. **Character Distribution**: Balanced distribution across vocabulary

### Training Pipeline

```python
# 1. Dataset Preparation
dataset = []
for file in directory:
    dataset.append([image_path, label_from_filename])
    
# 2. Vocabulary Extraction
vocab = set()
max_length = 0
for image_path, label in dataset:
    vocab.update(list(label))
    max_length = max(max_length, len(label))

# 3. Data Provider Setup
from mltu.dataProvider import DataProvider
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding

data_provider = DataProvider(
    dataset=dataset,
    batch_size=64,
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(width=200, height=50),
        LabelIndexer(vocab),
        LabelPadding(max_length, padding_value=len(vocab))
    ]
)

# 4. Train/Validation Split
train_data, val_data = data_provider.split(split=0.9)

# 5. Augmentation (Training Only)
train_data.augmentors = [
    RandomBrightness(),
    RandomRotate(angle=5),
    RandomErodeDilate()
]

# 6. Model Architecture
model = train_model(
    input_dim=(50, 200, 3),
    output_dim=len(vocab)
)

# 7. Compilation
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CTCloss(),
    metrics=[CWERMetric()]
)

# 8. Training with Callbacks
model.fit(
    train_data,
    validation_data=val_data,
    epochs=1000,
    callbacks=[
        EarlyStopping(monitor='val_CER', patience=40),
        ModelCheckpoint(monitor='val_CER', save_best_only=True),
        ReduceLROnPlateau(monitor='val_CER', factor=0.9, patience=20),
        TensorBoard(log_dir='logs/'),
        Model2onnx(output_path='model.onnx')
    ]
)
```

### Model Architecture Code

```python
def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")
    
    # Normalization
    x = layers.Lambda(lambda x: x / 255)(inputs)
    
    # CNN Feature Extraction (Residual Blocks)
    x = residual_block(x, 16, activation, skip_conv=True, strides=1, dropout=dropout)
    x = residual_block(x, 16, activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 32, activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 64, activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 64, activation, skip_conv=False, strides=1, dropout=dropout)
    
    # Reshape for RNN
    squeezed = layers.Reshape((x.shape[-3] * x.shape[-2], x.shape[-1]))(x)
    
    # Bidirectional LSTM
    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)
    
    # Output Layer (vocab_size + 1 for CTC blank token)
    output = layers.Dense(output_dim + 1, activation='softmax')(blstm)
    
    return Model(inputs=inputs, outputs=output)
```

---

### Legitimate Use Cases

-  **Internal Testing**: QA automation for government portal testing
-  **Accessibility**: Tools for users with visual impairments
-  **Research**: Academic study of CAPTCHA effectiveness
-  **Authorized Automation**: Compliance workflows with explicit permission

---

##  Acknowledgments

### Libraries & Frameworks

- **[MLTU Library](https://github.com/pythonlessons/mltu)** - Machine Learning Training Utilities for simplified OCR pipeline
- **TensorFlow/Keras** - Deep learning framework for model training
- **ONNX Runtime** - Cross-platform inference optimization
- **OpenCV** - Image processing and computer vision
- **Flask-RESTx** - RESTful API framework with Swagger documentation

### Theoretical Foundation

- **[Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)** - Alex Graves et al., ICML 2006
- **[Deep Residual Learning](https://arxiv.org/abs/1512.03385)** - He et al., CVPR 2016
- **[LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)** - Hochreiter & Schmidhuber, 1997

### Inspiration

This implementation is based on the captcha-to-text OCR tutorial by **PyLessons**, adapted and extended for production deployment with multiple CAPTCHA types and REST API integration.

---
