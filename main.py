import os
import torch
from models import CNN, ViT
from trainers import SimCLRTrainer, DINOTrainer
from datasets import get_mnist_loaders, get_cifar10_loaders, get_imagenet_loaders, get_imagenette_loaders
from torch.utils.data import random_split, DataLoader

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model = ViT(
        chw = (3, 160, 160),
        n_patches = 8,
        n_blocks = 6,
        hidden_d = 512,
        n_heads = 8,
        num_classes = 1000,
    ).to('cuda')

    print(model.n_heads)

    # Get data loaders
    train_loader, test_loader = get_imagenette_loaders(batch_size=128, num_workers=4)
    
    # Split the dataset into two parts
    # M = 0.99   # Proportion of data to use for pretraining
    # split = [M, 1 - M] if M < 1 else [len(train_loader.dataset) - M, M]
    # pretrain_dataset, finetune_dataset = random_split(train_loader.dataset, split)

    # # Create data loaders for each subset
    # pretrain_loader = DataLoader(
    #     pretrain_dataset, 
    #     batch_size=128, 
    #     shuffle=True, 
    #     num_workers=8
    # )
    # finetune_loader = DataLoader(
    #     finetune_dataset,
    #     batch_size=128,
    #     shuffle=True,
    #     num_workers=8
    # )
    
    # Initialize DINO trainer
    trainer = DINOTrainer(model)

    # # Train with DINO
    # print("Training with DINO...")
    # dino_trainer.train(
    #     train_loader=pretrain_loader,
    #     test_loader=test_loader,
    #     epochs=1,
    # )

    # Finetune DINO model
    print("\nFinetuning DINO model...")
    history = trainer.finetune(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1,
        patience=1,
        visualize_every = 20
    )



    for i in range(5):
        print(f"Image {i} class: {[test_loader.dataset[i][1]]}")
        for layer in range(model.n_blocks):
            for head in range(model.n_heads):
                model.visualize_attention(
                    images=test_loader.dataset[i][0].unsqueeze(0).to('cuda'),
                    layer_idx=layer,  # Transformer block
                    head_idx=head,    # Attention head
                    save_path=f'attention_map_{i}_{layer}_{head}.png'
                )

    # Print final results
    print("\nFinal Results:")
    print(f"DINO Final Test Accuracy: {dino_history['train_acc'][-1]:.2f}%")

if __name__ == '__main__':
    main()