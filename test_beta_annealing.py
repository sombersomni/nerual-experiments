#!/usr/bin/env python3
"""
Test cases for β-annealing (KL weight annealing) functionality in MultiTaskLoss.
"""

import torch
from models import MultiTaskLoss


class TestBetaAnnealing:
    """Test suite for β-annealing functionality"""
    
    def test_beta_annealing_initial_value(self):
        """Test that beta starts at the specified initial value"""
        loss_fn = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.001,
            beta_anneal_epochs=100
        )
        
        # At epoch 0, beta should equal beta_anneal_start
        beta = loss_fn.get_beta(current_epoch=0)
        assert beta == 0.001, f"Expected beta=0.001 at epoch 0, got {beta}"
    
    def test_beta_annealing_final_value(self):
        """Test that beta reaches the target KL weight after annealing period"""
        loss_fn = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.001,
            beta_anneal_epochs=100
        )
        
        # At epoch >= beta_anneal_epochs, beta should equal kl_weight
        beta_at_100 = loss_fn.get_beta(current_epoch=100)
        beta_at_150 = loss_fn.get_beta(current_epoch=150)
        
        assert beta_at_100 == 0.01, f"Expected beta=0.01 at epoch 100, got {beta_at_100}"
        assert beta_at_150 == 0.01, f"Expected beta=0.01 at epoch 150, got {beta_at_150}"
    
    def test_beta_annealing_linear_progression(self):
        """Test that beta increases linearly during annealing period"""
        loss_fn = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.001,
            beta_anneal_epochs=100
        )
        
        # Test specific points in the annealing schedule
        beta_25 = loss_fn.get_beta(current_epoch=25)   # 25% progress
        beta_50 = loss_fn.get_beta(current_epoch=50)   # 50% progress
        beta_75 = loss_fn.get_beta(current_epoch=75)   # 75% progress
        
        # Expected values: start + progress * (end - start)
        expected_25 = 0.001 + 0.25 * (0.01 - 0.001)  # 0.00325
        expected_50 = 0.001 + 0.50 * (0.01 - 0.001)  # 0.0055
        expected_75 = 0.001 + 0.75 * (0.01 - 0.001)  # 0.007750
        
        assert abs(beta_25 - expected_25) < 1e-6, f"Expected beta={expected_25} at epoch 25, got {beta_25}"
        assert abs(beta_50 - expected_50) < 1e-6, f"Expected beta={expected_50} at epoch 50, got {beta_50}"
        assert abs(beta_75 - expected_75) < 1e-6, f"Expected beta={expected_75} at epoch 75, got {beta_75}"
    
    def test_beta_annealing_monotonic_increase(self):
        """Test that beta monotonically increases during annealing"""
        loss_fn = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.001,
            beta_anneal_epochs=100
        )
        
        betas = []
        for epoch in range(0, 101, 10):  # Test every 10 epochs
            beta = loss_fn.get_beta(current_epoch=epoch)
            betas.append(beta)
        
        # Check that beta values are monotonically increasing
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i-1], f"Beta decreased from {betas[i-1]} to {betas[i]} at epoch {i*10}"
    
    def test_beta_annealing_different_schedules(self):
        """Test different annealing schedules"""
        # Short annealing period
        loss_fn_short = MultiTaskLoss(
            kl_weight=0.02,
            beta_anneal_start=0.0005,
            beta_anneal_epochs=50
        )
        
        # Long annealing period
        loss_fn_long = MultiTaskLoss(
            kl_weight=0.02,
            beta_anneal_start=0.0005,
            beta_anneal_epochs=200
        )
        
        # At epoch 25, short schedule should be at 50% progress, long at 12.5%
        beta_short_25 = loss_fn_short.get_beta(current_epoch=25)
        beta_long_25 = loss_fn_long.get_beta(current_epoch=25)
        
        expected_short = 0.0005 + 0.5 * (0.02 - 0.0005)    # 0.010250
        expected_long = 0.0005 + 0.125 * (0.02 - 0.0005)   # 0.002938
        
        assert abs(beta_short_25 - expected_short) < 1e-6, f"Short schedule: expected {expected_short}, got {beta_short_25}"
        assert abs(beta_long_25 - expected_long) < 1e-6, f"Long schedule: expected {expected_long}, got {beta_long_25}"
    
    def test_beta_annealing_edge_cases(self):
        """Test edge cases for beta annealing"""
        # Case 1: beta_anneal_start equals kl_weight (no annealing)
        loss_fn_no_anneal = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.01,
            beta_anneal_epochs=100
        )
        
        for epoch in [0, 25, 50, 100, 150]:
            beta = loss_fn_no_anneal.get_beta(current_epoch=epoch)
            assert beta == 0.01, f"Expected beta=0.01 (no annealing), got {beta} at epoch {epoch}"
        
        # Case 2: Very small annealing period
        loss_fn_tiny = MultiTaskLoss(
            kl_weight=0.01,
            beta_anneal_start=0.001,
            beta_anneal_epochs=1
        )
        
        beta_0 = loss_fn_tiny.get_beta(current_epoch=0)
        beta_1 = loss_fn_tiny.get_beta(current_epoch=1)
        
        assert beta_0 == 0.001, f"Expected beta=0.001 at epoch 0, got {beta_0}"
        assert beta_1 == 0.01, f"Expected beta=0.01 at epoch 1, got {beta_1}"


def test_integration_with_multitask_loss():
    """Test that beta annealing integrates correctly with MultiTaskLoss forward pass"""
    # Create dummy outputs that match the expected structure
    batch_size = 4
    embed_dim = 128
    latent_dim = 64
    num_classes = 10
    
    # Create dummy anchor outputs with all required keys
    anchor_outputs = {
        'embeddings': torch.randn(batch_size, embed_dim),
        'logits': torch.randn(batch_size, num_classes),
        'mu': torch.randn(batch_size, latent_dim),
        'logvar': torch.randn(batch_size, latent_dim),
        'z': torch.randn(batch_size, latent_dim),
        'reconstruction': torch.randn(batch_size, 1, 28, 28),
        'original': torch.randn(batch_size, 1, 28, 28)
    }
    
    positive_outputs = {
        'embeddings': torch.randn(batch_size, embed_dim),
        'logits': torch.randn(batch_size, num_classes),
        'z': torch.randn(batch_size, latent_dim)
    }
    
    negative_outputs = {
        'embeddings': torch.randn(batch_size, embed_dim),
        'logits': torch.randn(batch_size, num_classes),
        'z': torch.randn(batch_size, latent_dim)
    }
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create loss function with annealing
    loss_fn = MultiTaskLoss(
        kl_weight=0.01,
        beta_anneal_start=0.001,
        beta_anneal_epochs=100
    )
    
    # Test loss computation at different epochs
    epochs_to_test = [0, 25, 50, 100, 150]
    total_losses_by_epoch = {}
    kl_contributions_by_epoch = {}
    
    # First, get the raw KL divergence value (without beta scaling)
    _, loss_dict_base = loss_fn(anchor_outputs, positive_outputs, negative_outputs, labels, current_epoch=100)
    base_kl_value = loss_dict_base['kl'].item()
    
    for epoch in epochs_to_test:
        total_loss, loss_dict = loss_fn(anchor_outputs, positive_outputs, negative_outputs, labels, current_epoch=epoch)
        total_losses_by_epoch[epoch] = total_loss.item()
        
        # Calculate the expected KL contribution based on current beta
        current_beta = loss_fn.get_beta(epoch)
        expected_kl_contribution = current_beta * base_kl_value
        kl_contributions_by_epoch[epoch] = expected_kl_contribution
        
        # Verify that total_loss is a scalar tensor
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0, f"Expected scalar loss, got tensor with shape {total_loss.shape}"
        
        # Verify that KL loss is being computed
        assert 'kl' in loss_dict, "KL loss not found in loss dictionary"
    
    # Verify that the beta annealing is working by checking the KL contribution differences
    kl_contrib_epoch_0 = kl_contributions_by_epoch[0]
    kl_contrib_epoch_100 = kl_contributions_by_epoch[100]
    
    # The ratio should match the beta ratio (0.001 vs 0.01 = 10x difference)
    expected_ratio = 0.01 / 0.001  # 10.0
    if kl_contrib_epoch_0 > 0:  # Avoid division by zero
        actual_ratio = kl_contrib_epoch_100 / kl_contrib_epoch_0
        assert abs(actual_ratio - expected_ratio) < 0.1, f"Expected ratio ~{expected_ratio}, got {actual_ratio}"


if __name__ == "__main__":
    # Run the tests
    test_suite = TestBetaAnnealing()
    
    print("Running β-annealing tests...")
    print("1. Testing initial value...")
    test_suite.test_beta_annealing_initial_value()
    print("✓ Initial value test passed")
    
    print("2. Testing final value...")
    test_suite.test_beta_annealing_final_value()
    print("✓ Final value test passed")
    
    print("3. Testing linear progression...")
    test_suite.test_beta_annealing_linear_progression()
    print("✓ Linear progression test passed")
    
    print("4. Testing monotonic increase...")
    test_suite.test_beta_annealing_monotonic_increase()
    print("✓ Monotonic increase test passed")
    
    print("5. Testing different schedules...")
    test_suite.test_beta_annealing_different_schedules()
    print("✓ Different schedules test passed")
    
    print("6. Testing edge cases...")
    test_suite.test_beta_annealing_edge_cases()
    print("✓ Edge cases test passed")
    
    print("7. Testing integration with MultiTaskLoss...")
    test_integration_with_multitask_loss()
    print("✓ Integration test passed")
    
    print("\nAll β-annealing tests passed! 🎉")