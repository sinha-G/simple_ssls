# main.py
import torch
from models import CNN
from trainers import SimCLRTrainer, DINOTrainer
from datasets import get_mnist_loaders, get_cifar10_loaders
from torch.utils.data import random_split, DataLoader

def main():
    # Initialize model and set M
    model = CNN(
        use_dropout=True, 
        dropout_rate=0.3, 
        use_projection_head=True,
        input_channels=3
    )
    model = model.to('cuda')

    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Split the dataset into two parts
    M = 100
    split = [1 - M, M] if M < 1 else [len(train_loader.dataset) - M, M]
    pretrain_dataset, finetune_dataset = random_split(train_loader.dataset, split)

    # Create data loaders for each subset
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=0
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize DINO trainer
    dino_trainer = DINOTrainer(
        model,
        lr=0.00005,
        momentum_teacher=0.996,
        momentum_center=0.9,
        temp_student=0.1,
        temp_teacher=0.04,
        n_local_views=6
    )

    # Train with DINO
    print("Training with DINO...")
    train_losses, test_accs = dino_trainer.train(
        train_loader=pretrain_loader,
        test_loader=test_loader,
        epochs=100,
        evaluate_every=1
    )

    # Initialize supervised baseline
    baseline_model = CNN(use_dropout=True, dropout_rate=0.3, use_projection_head=True)
    baseline_trainer = DINOTrainer(baseline_model)

    # Train baseline directly on labeled data
    print("\nTraining baseline...")
    baseline_losses, baseline_accs = baseline_trainer.train(
        train_loader=finetune_loader,
        test_loader=test_loader,
        epochs=100,
        evaluate_every=5
    )

    # Print final results
    print("\nFinal Results:")
    print(f"DINO Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Baseline Test Accuracy: {baseline_accs[-1]:.2f}%")

if __name__ == '__main__':
    main()