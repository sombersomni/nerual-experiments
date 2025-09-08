#!/usr/bin/env python3
"""
Visualize β-annealing schedule for KL weight.
"""

from models import MultiTaskLoss


def visualize_beta_annealing():
    """Visualize the β-annealing schedule"""
    
    # Create loss function with annealing
    loss_fn = MultiTaskLoss(
        kl_weight=0.01,
        beta_anneal_start=0.001,
        beta_anneal_epochs=100
    )
    
    print("β-Annealing Schedule Visualization")
    print("=" * 40)
    print(f"{'Epoch':<8} {'Beta (KL Weight)':<15} {'Progress':<10}")
    print("-" * 40)
    
    # Show annealing schedule at key points
    test_epochs = [0, 10, 25, 50, 75, 90, 100, 110, 150]
    
    for epoch in test_epochs:
        beta = loss_fn.get_beta(epoch)
        if epoch < loss_fn.beta_anneal_epochs:
            progress = f"{epoch/loss_fn.beta_anneal_epochs*100:.1f}%"
        else:
            progress = "Complete"
        
        print(f"{epoch:<8} {beta:<15.6f} {progress:<10}")
    
    print("\nAnnealing Details:")
    print(f"  Start value: {loss_fn.beta_anneal_start}")
    print(f"  Final value: {loss_fn.kl_weight}")
    print(f"  Annealing period: {loss_fn.beta_anneal_epochs} epochs")
    print(f"  Multiplier: {loss_fn.kl_weight/loss_fn.beta_anneal_start:.1f}x increase")


if __name__ == "__main__":
    visualize_beta_annealing()