# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses a Python virtual environment located at `~/nerual-ai-venv`. Always activate the virtual environment before running Python commands:

```bash
source  ~/nerual-ai-venv/bin/activate
python -m <command>  # Use python -m to ensure running through venv
```

## Key Commands

### Training Models
```bash
# Train with specific config (required)
python run.py --config configs/<dataset>.yaml --epochs <num_epochs>

# Example commands:
python run.py --config configs/mnist.yaml --epochs 10
python run.py --config configs/fashionmnist.yaml --epochs 15
python run.py --config configs/cifar10.yaml --epochs 20

# Override random seed
python run.py --config configs/mnist.yaml --epochs 5 --seed 123
```

### Available Datasets
- `mnist` - MNIST handwritten digits (28x28, grayscale)
- `fashionmnist` - Fashion-MNIST clothing images (28x28, grayscale) 
- `cifar10` - CIFAR-10 natural images (32x32, RGB)

## Architecture Overview

This is a neural network experimentation codebase focused on learning image embeddings using triplet loss combined with classification.

### Core Components

1. **ImageTransformer** (`run.py:76-150`) - Vision Transformer architecture that:
   - Splits images into patches and applies patch embedding
   - Uses positional encoding and transformer layers
   - Outputs both embeddings (for triplet loss) and classification logits

2. **TripletDataset** (`dataset.py:9-61`) - Custom dataset that creates triplets:
   - Anchor: Base image
   - Positive: Same class as anchor
   - Negative: Different class from anchor
   - Supports data augmentation (rotation, affine transforms, perspective)

3. **TripletLoss** (`run.py:152-161`) - Custom loss function with configurable margin

### Training Process

The model is trained with a combined loss function:
- **Triplet Loss**: Learns embeddings where same-class samples are closer than different-class samples
- **Classification Loss**: Standard cross-entropy for class prediction
- Final loss = triplet_loss + (classifier_weight × classification_loss)

### Configuration System

All hyperparameters are defined in YAML config files in `configs/`:
- Model architecture (patch_size, embed_dim, num_heads, num_layers)
- Training parameters (learning_rate, weight_decay, scheduler settings)
- Dataset settings (batch_size, num_workers)
- Logging options (visualization, sample frequency)

### Output Files

Training generates:
- Model checkpoints: `models/image_transformer_<dataset>_<epochs>epochs_<accuracy>acc_<timestamp>.pth`
- Best model: `models/best_<dataset>_model.pth`
- Training plots: `training_progress_<dataset>.png`
- Embedding visualizations: `embeddings_tsne.png`
- Triplet samples: `triplet_samples_<dataset>.png`

### Model Loading

Use `load_model()` function in `run.py:295-331` to load saved models:
```python
model, checkpoint = load_model('models/best_mnist_model.pth', device)
```

## Development Workflow

1. Modify config files in `configs/` for hyperparameter tuning
2. Run training with `python run.py --config <config_file> --epochs <n>`
3. Models are automatically saved with performance metrics
4. Visualizations help assess training quality and embedding clustering
5. Best models are tracked automatically for each dataset