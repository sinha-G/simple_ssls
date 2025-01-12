import os
import torch
from models import CNN, ViT
from trainers import SimCLRTrainer, DINOTrainer
from datasets import get_mnist_loaders, get_cifar10_loaders, get_imagenet_loaders
from torch.utils.data import random_split, DataLoader

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Initialize model and set M
    # model = CNN(
    #     use_dropout=True, 
    #     dropout_rate=0.3, 
    #     use_projection_head=True,
    #     input_channels=3
    # )
    model = ViT(
        chw = (3, 224, 224),
        n_patches = 14,
        n_blocks = 6,
        hidden_d = 256,
        n_heads = 8,
        num_classes = 1000,
        dropout = 0.1
    )
    model = model.to('cuda')

    # Get data loaders
    print("Loading data...")
    train_loader, test_loader = get_imagenet_loaders(batch_size=128, num_workers=8)
    
    # Split the dataset into two parts
    print("Splitting dataset...")
    M = 0.99   # Proportion of data to use for pretraining
    split = [M, 1 - M] if M < 1 else [len(train_loader.dataset) - M, M]
    pretrain_dataset, finetune_dataset = random_split(train_loader.dataset, split)

    # Create data loaders for each subset
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=8
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8
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
    dino_trainer.train(
        train_loader=pretrain_loader,
        test_loader=test_loader,
        epochs=1,
    )

    # Finetune DINO model
    print("\nFinetuning DINO model...")
    dino_history = dino_trainer.finetune(
        train_loader=finetune_loader,
        test_loader=test_loader,
        epochs=10,
        lr=0.0001,
        patience=10,
        evaluate_every=1
    )

    # Initialize supervised baseline
    baseline_model = ViT(
        chw = (3, 224, 224),
        n_patches = 14,
        n_blocks = 6,
        hidden_d = 256,
        n_heads = 8,
        num_classes = 1000,
        dropout = 0.1
    )
    baseline_trainer = DINOTrainer(baseline_model)

    # Train baseline directly on labeled data
    print("\nTraining baseline...")
    baseline_history = baseline_trainer.finetune(
        train_loader=finetune_loader,
        test_loader=test_loader,
        epochs=10,
        lr=0.0001,
        patience=10,
        evaluate_every=1
    )

    # Print final results
    print("\nFinal Results:")
    print(f"DINO Final Test Accuracy: {dino_history['train_acc'][-1]:.2f}%")
    print(f"Baseline Test Accuracy: {baseline_history['train_acc']:.2f}%")

if __name__ == '__main__':
    main()