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

This is a neural network experimentation codebase focused on learning image embeddings using triplet loss combined with classification and reconstruction tasks through a multi-task learning framework built on PyTorch Lightning.

### Core Components

1. **MultiTaskImageTransformer** (`models.py:74-189`) - Advanced Vision Transformer with VAE components:
   - Splits images into patches and applies patch embedding
   - Uses positional encoding and transformer encoder layers
   - Includes VAE latent space with mu/logvar heads and reparameterization trick
   - Embedding head operates on latent samples for triplet loss
   - Classification head predicts classes from embeddings
   - VAE decoder reconstructs images from latent space

2. **ImageTransformer** (`models.py:192-199`) - Backward compatibility wrapper around MultiTaskImageTransformer for simple triplet + classification tasks

3. **TripletDataset** (`dataset.py:9-61`) - Custom dataset that creates triplets:
   - Anchor: Base image
   - Positive: Same class as anchor
   - Negative: Different class from anchor
   - Supports data augmentation (rotation, affine transforms, perspective)

4. **Loss Functions** (`models.py:201-315`):
   - **TripletLoss**: Configurable margin-based triplet loss
   - **KLDivergenceLoss**: VAE regularization loss
   - **ReconstructionLoss**: MSE/BCE loss for image reconstruction
   - **MultiTaskLoss**: Combines all losses with configurable weights

### Training Process

The model supports two training modes:

#### Simple Mode (ImageTransformerLightning)
- **Triplet Loss**: Learns embeddings where same-class samples are closer than different-class samples
- **Classification Loss**: Standard cross-entropy for class prediction
- Final loss = triplet_loss + (classifier_weight × classification_loss)

#### Multi-Task Mode (MultiTaskImageTransformerLightning)
- **Triplet Loss**: On embeddings derived from VAE latent space
- **Classification Loss**: Standard cross-entropy on predicted classes
- **Reconstruction Loss**: MSE loss between original and reconstructed images
- **KL Divergence Loss**: VAE regularization to keep latent space well-behaved
- Final loss = triplet_weight × triplet_loss + classification_weight × classification_loss + reconstruction_weight × reconstruction_loss + kl_weight × kl_loss

### PyTorch Lightning Training Framework

The codebase uses PyTorch Lightning for streamlined training with automatic checkpointing, logging, and distributed training support:

#### Lightning Models (`models.py:317-637`)
- **ImageTransformerLightning**: Simple triplet + classification training
- **MultiTaskImageTransformerLightning**: Full multi-task learning with VAE components

#### Training Features
- Automatic model checkpointing with best model tracking
- Early stopping based on validation accuracy
- TensorBoard logging with loss curves and reconstruction visualizations
- Automatic GPU/CPU detection and optimization
- Configurable learning rate schedulers (StepLR, CosineAnnealingLR, ExponentialLR)

#### Trainer Configuration (`run.py:77-126`)
- Callbacks for checkpointing and early stopping
- TensorBoard logger with experiment versioning
- Progress bars and model summaries
- Deterministic training support

### Configuration System

All hyperparameters are defined in YAML config files in `configs/`:
- **Model architecture**: patch_size, embed_dim, num_heads, num_layers, latent_dim
- **Training parameters**: learning_rate, weight_decay, scheduler settings, loss weights
- **Multi-task settings**: use_multitask_loss flag and individual loss weights
- **Dataset settings**: batch_size, num_workers
- **Logging options**: visualization flags, sample frequency, TensorBoard settings

### Output Files

PyTorch Lightning training generates:
- **Lightning checkpoints**: `lightning_checkpoints/<dataset>/epoch=XX-val_accuracy=X.XX.ckpt`
- **TensorBoard logs**: `img_tasks/<dataset>/experiment/version_X/`
- **Embedding visualizations**: `embeddings_tsne.png` (t-SNE plots of learned embeddings)
- **Triplet samples**: `triplet_samples_<dataset>.png` (visualization of anchor/positive/negative triplets)
- **Reconstruction images**: Logged to TensorBoard during validation (multi-task mode only)

### Model Loading

#### Lightning Checkpoints (Preferred)
Use `load_lightning_model()` function in `run.py:173-198` to load Lightning checkpoints:
```python
# Auto-detect model type
model = load_lightning_model('lightning_checkpoints/mnist/best.ckpt')

# Specify model type
model = load_lightning_model('path/to/checkpoint.ckpt', MultiTaskImageTransformerLightning)
```

#### Legacy Model Loading (Backward Compatibility)
Use `load_model()` function in `run.py:200-237` for old .pth files:
```python
model, checkpoint = load_model('models/best_mnist_model.pth', device)
```

## Development Workflow

1. **Configure training**: Modify YAML config files in `configs/` for hyperparameter tuning
   - Set `use_multitask_loss: true` for multi-task VAE training
   - Adjust loss weights (triplet_weight, classification_weight, reconstruction_weight, kl_weight)
   - Configure model architecture and training parameters

2. **Run training**: `python run.py --config <config_file> --epochs <n>`
   - Lightning handles checkpointing, logging, and early stopping automatically
   - Monitor progress with TensorBoard: `tensorboard --logdir=img_tasks/<dataset>`

3. **Model evaluation**: 
   - Best models are automatically saved based on validation accuracy
   - Embedding visualizations and reconstruction quality are logged
   - Test accuracy is reported at the end of training

4. **Model analysis**:
   - View t-SNE embeddings to assess learned representations
   - Check reconstruction quality in TensorBoard (multi-task mode)
   - Compare triplet loss convergence and classification accuracy

5. **Experiment tracking**: All experiments are versioned and logged with TensorBoard integration