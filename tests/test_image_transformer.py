import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import ImageTransformer, MultiTaskImageTransformer, TripletLoss, MultiTaskLoss, KLDivergenceLoss, ReconstructionLoss


class TestImageTransformer:
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.input_size = 28
        self.num_channels = 1
        self.num_classes = 10
        
        # Create model
        self.model = ImageTransformer(
            input_size=self.input_size,
            num_channels=self.num_channels,
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Create sample data
        self.sample_data = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        self.sample_labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
    
    def test_model_forward_pass(self):
        """Test that the model forward pass produces correct output shapes"""
        embeddings, logits = self.model(self.sample_data)
        
        # Check output shapes
        assert embeddings.shape == (self.batch_size, 128), f"Expected embeddings shape {(self.batch_size, 128)}, got {embeddings.shape}"
        assert logits.shape == (self.batch_size, self.num_classes), f"Expected logits shape {(self.batch_size, self.num_classes)}, got {logits.shape}"
        
        # Check that outputs are finite
        assert torch.isfinite(embeddings).all(), "Embeddings contain non-finite values"
        assert torch.isfinite(logits).all(), "Logits contain non-finite values"


class TestTripletLoss:
    def setup_method(self):
        """Setup test fixtures for triplet loss"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.embedding_dim = 128
        self.margin = 2.0
        
        self.triplet_loss = TripletLoss(margin=self.margin)
        
        # Create sample embeddings
        self.anchor = torch.randn(self.batch_size, self.embedding_dim).to(self.device)
        self.positive = torch.randn(self.batch_size, self.embedding_dim).to(self.device)
        self.negative = torch.randn(self.batch_size, self.embedding_dim).to(self.device)
    
    def test_triplet_loss_computation(self):
        """Test triplet loss computation"""
        loss = self.triplet_loss(self.anchor, self.positive, self.negative)
        
        # Loss should be a scalar
        assert loss.dim() == 0, f"Expected scalar loss, got tensor with {loss.dim()} dimensions"
        
        # Loss should be non-negative
        assert loss.item() >= 0, f"Expected non-negative loss, got {loss.item()}"
        
        # Loss should be finite
        assert torch.isfinite(loss), "Loss is not finite"
    
    def test_triplet_loss_margin_effect(self):
        """Test that margin affects triplet loss correctly"""
        # Create controlled embeddings where positive is closer than negative
        anchor = torch.zeros(1, self.embedding_dim).to(self.device)
        positive = torch.ones(1, self.embedding_dim).to(self.device) * 0.1  # Distance = 0.1 * sqrt(embedding_dim)
        negative = torch.ones(1, self.embedding_dim).to(self.device) * 1.0   # Distance = 1.0 * sqrt(embedding_dim)
        
        # Test with small margin (should have smaller loss since constraint is more satisfied)
        small_margin_loss = TripletLoss(margin=0.1)
        loss_small = small_margin_loss(anchor, positive, negative)
        
        # Test with large margin (should have larger loss since constraint is less satisfied)
        large_margin_loss = TripletLoss(margin=5.0)
        loss_large = large_margin_loss(anchor, positive, negative)
        
        # Large margin should produce larger loss since the constraint is harder to satisfy
        assert loss_large >= loss_small, f"Large margin should produce larger or equal loss, got small={loss_small.item():.6f}, large={loss_large.item():.6f}"
    
    def test_triplet_loss_zero_when_satisfied(self):
        """Test that triplet loss is zero when margin constraint is satisfied"""
        # Create embeddings where negative is much farther than positive + margin
        anchor = torch.zeros(1, self.embedding_dim).to(self.device)
        positive = torch.ones(1, self.embedding_dim).to(self.device) * 0.1  # Close to anchor
        negative = torch.ones(1, self.embedding_dim).to(self.device) * 10.0  # Far from anchor
        
        loss = self.triplet_loss(anchor, positive, negative)
        
        # Loss should be zero or very close to zero since constraint is satisfied
        assert loss.item() < 1e-6, f"Expected near-zero loss when constraint is satisfied, got {loss.item()}"


class TestMultiTaskLoss:
    def setup_method(self):
        """Setup test fixtures for multi-task loss"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.embedding_dim = 128
        self.num_classes = 10
        self.input_size = 28
        self.num_channels = 1
        
        # Create model
        self.model = MultiTaskImageTransformer(
            input_size=self.input_size,
            num_channels=self.num_channels,
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_classes=self.num_classes,
            latent_dim=32
        ).to(self.device)
        
        # Create multi-task loss
        self.multitask_loss = MultiTaskLoss(
            triplet_weight=1.0,
            classification_weight=0.5,
            reconstruction_weight=1.0,
            kl_weight=0.01,
            margin=2.0
        )
        
        # Sample data
        self.sample_data = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        self.sample_labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
    
    def test_multitask_forward_pass(self):
        """Test multi-task model forward pass produces correct outputs"""
        outputs = self.model(self.sample_data, reconstruct=True)
        
        # Check all required keys are present
        required_keys = ['embeddings', 'logits', 'mu', 'logvar', 'z', 'reconstruction', 'original']
        for key in required_keys:
            assert key in outputs, f"Missing key {key} in model outputs"
        
        # Check shapes
        assert outputs['embeddings'].shape == (self.batch_size, 128)
        assert outputs['logits'].shape == (self.batch_size, self.num_classes)
        assert outputs['mu'].shape == (self.batch_size, 32)  # latent_dim=32
        assert outputs['logvar'].shape == (self.batch_size, 32)
        assert outputs['z'].shape == (self.batch_size, 32)
        assert outputs['reconstruction'].shape == (self.batch_size, self.num_channels, 32, 32)  # Decoder outputs 32x32
        assert outputs['original'].shape == (self.batch_size, self.num_channels, self.input_size, self.input_size)
    
    def test_multitask_loss_computation(self):
        """Test multi-task loss computation"""
        # Get model outputs
        anchor_outputs = self.model(self.sample_data, reconstruct=True)
        positive_outputs = self.model(self.sample_data, reconstruct=False)  # Different samples
        negative_outputs = self.model(self.sample_data, reconstruct=False)
        
        # Compute loss (with current_epoch=0 for testing)
        total_loss, losses = self.multitask_loss(anchor_outputs, positive_outputs, negative_outputs, self.sample_labels, current_epoch=0)
        
        # Check that all loss components are present
        expected_loss_keys = ['triplet', 'classification', 'reconstruction', 'kl', 'total']
        for key in expected_loss_keys:
            assert key in losses, f"Missing loss component: {key}"
        
        # Check that all losses are scalars and finite
        for key, loss_val in losses.items():
            assert loss_val.dim() == 0, f"{key} loss should be scalar"
            assert torch.isfinite(loss_val), f"{key} loss is not finite"
            if key != 'total':  # Individual losses should be non-negative
                assert loss_val.item() >= 0, f"{key} loss should be non-negative"
        
        # Check that total loss equals weighted sum (using annealed beta)
        current_epoch = 0
        current_beta = self.multitask_loss.get_beta(current_epoch)
        expected_total = (
            self.multitask_loss.triplet_weight * losses['triplet'] +
            self.multitask_loss.classification_weight * losses['classification'] +
            self.multitask_loss.reconstruction_weight * losses['reconstruction'] +
            current_beta * losses['kl']
        )
        
        assert torch.allclose(losses['total'], expected_total, rtol=1e-5), "Total loss doesn't match weighted sum"
        assert torch.allclose(total_loss, losses['total'], rtol=1e-5), "Returned total_loss doesn't match losses['total']"


class TestKLDivergenceLoss:
    def setup_method(self):
        """Setup test fixtures for KL divergence loss"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.latent_dim = 32
        
        self.kl_loss = KLDivergenceLoss()
    
    def test_kl_loss_computation(self):
        """Test KL divergence loss computation"""
        mu = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        logvar = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        
        loss = self.kl_loss(mu, logvar)
        
        # Loss should be scalar and finite
        assert loss.dim() == 0, "KL loss should be scalar"
        assert torch.isfinite(loss), "KL loss should be finite"
        
        # KL divergence should be non-negative
        assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
    
    def test_kl_loss_zero_for_standard_normal(self):
        """Test that KL loss is zero when mu=0 and logvar=0 (standard normal)"""
        mu = torch.zeros(self.batch_size, self.latent_dim).to(self.device)
        logvar = torch.zeros(self.batch_size, self.latent_dim).to(self.device)
        
        loss = self.kl_loss(mu, logvar)
        
        # Should be very close to zero
        assert abs(loss.item()) < 1e-6, f"KL loss should be zero for standard normal, got {loss.item()}"


class TestReconstructionLoss:
    def setup_method(self):
        """Setup test fixtures for reconstruction loss"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        
        self.recon_loss_mse = ReconstructionLoss('mse')
        self.recon_loss_bce = ReconstructionLoss('bce')
    
    def test_reconstruction_loss_mse(self):
        """Test MSE reconstruction loss"""
        # Test with matching dimensions
        original = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        reconstruction = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        
        loss = self.recon_loss_mse(reconstruction, original)
        
        assert loss.dim() == 0, "Reconstruction loss should be scalar"
        assert torch.isfinite(loss), "Reconstruction loss should be finite"
        assert loss.item() >= 0, "MSE loss should be non-negative"
    
    def test_reconstruction_loss_bce(self):
        """Test BCE reconstruction loss"""
        # BCE requires values in [0, 1]
        original = torch.rand(self.batch_size, 1, 28, 28).to(self.device)
        reconstruction = torch.rand(self.batch_size, 1, 28, 28).to(self.device)
        
        loss = self.recon_loss_bce(reconstruction, original)
        
        assert loss.dim() == 0, "Reconstruction loss should be scalar"
        assert torch.isfinite(loss), "Reconstruction loss should be finite"
        assert loss.item() >= 0, "BCE loss should be non-negative"
    
    def test_reconstruction_loss_size_mismatch_handling(self):
        """Test that reconstruction loss handles size mismatches correctly"""
        # Original as flattened MNIST (784 = 28*28)
        original = torch.randn(self.batch_size, 784).to(self.device)
        # Reconstruction as 32x32 (decoder output size)
        reconstruction = torch.randn(self.batch_size, 1, 32, 32).to(self.device)
        
        loss = self.recon_loss_mse(reconstruction, original)
        
        # Should handle the size mismatch and compute loss
        assert loss.dim() == 0, "Reconstruction loss should be scalar"
        assert torch.isfinite(loss), "Reconstruction loss should be finite"
        assert loss.item() >= 0, "MSE loss should be non-negative"


class TestLossIntegration:
    def setup_method(self):
        """Setup integration test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.input_size = 28
        self.num_channels = 1
        self.num_classes = 10
        
        self.model = ImageTransformer(
            input_size=self.input_size,
            num_channels=self.num_channels,
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_classes=self.num_classes
        ).to(self.device)
        
        self.triplet_loss = TripletLoss(margin=2.0)
        self.classification_loss = nn.CrossEntropyLoss()
    
    def test_combined_loss_computation(self):
        """Test that combined triplet + classification loss works correctly"""
        # Create triplet data
        anchor = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        positive = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        negative = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Get model outputs
        anchor_emb, anchor_logits = self.model(anchor)
        positive_emb, _ = self.model(positive)
        negative_emb, _ = self.model(negative)
        
        # Compute individual losses
        triplet_l = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        class_l = self.classification_loss(anchor_logits, labels)
        
        # Compute combined loss
        classifier_weight = 0.5
        total_loss = triplet_l + classifier_weight * class_l
        
        # Verify all losses are valid
        assert torch.isfinite(triplet_l), "Triplet loss should be finite"
        assert torch.isfinite(class_l), "Classification loss should be finite"
        assert torch.isfinite(total_loss), "Total loss should be finite"
        
        assert triplet_l.item() >= 0, "Triplet loss should be non-negative"
        assert class_l.item() >= 0, "Classification loss should be non-negative"
        
        # Test that loss enables gradient computation
        self.model.zero_grad()  # Clear any existing gradients
        total_loss.backward()
        
        # Check that gradients were computed for parameters that require grad
        grad_params = []
        no_grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params.append(name)
                else:
                    no_grad_params.append(name)
        
        # At least some parameters should have gradients
        assert len(grad_params) > 0, f"No gradients computed. Params without grad: {no_grad_params[:5]}"
    
    def test_loss_backward_compatibility(self):
        """Test that ImageTransformer loss computation matches expected behavior"""
        # This test ensures the model works with the existing training code
        data = torch.randn(self.batch_size, self.input_size * self.input_size).to(self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        embeddings, logits = self.model(data)
        
        # Test classification loss
        class_loss = self.classification_loss(logits, labels)
        
        # Test that we can compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        assert torch.isfinite(class_loss), "Classification loss should be finite"
        assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, got {accuracy}"


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    import traceback
    
    test_classes = [
        TestImageTransformer,
        TestTripletLoss,
        TestMultiTaskLoss,
        TestKLDivergenceLoss,
        TestReconstructionLoss,
        TestLossIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}:")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                traceback.print_exc()
    
    print(f"\nTest Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed!")
        exit(1)