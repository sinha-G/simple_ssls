# main.py
import torch
from models import CNN
from trainers import SimCLRTrainer
from datasets import get_mnist_loaders
from torch.utils.data import random_split, DataLoader

def main():
    # Initialize model and set M
    model = CNN(use_dropout=True, dropout_rate=0.3, use_projection_head=True)
    model = model.to('cuda')

    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Split the dataset into two parts
    M = 20
    split = [1 - M, M] if M < 1 else [len(train_loader.dataset) - M, M]
    simclr_dataset, finetune_dataset = random_split(train_loader.dataset, split)

    # Create data loaders for each subset
    simclr_loader = DataLoader(simclr_dataset, batch_size=2048, shuffle=True, num_workers=0)
    finetune_loader = DataLoader(finetune_dataset, batch_size=1024, shuffle=True, num_workers=0)
    
    # Initialize SimCLR trainer 
    simclr_trainer = SimCLRTrainer(model)

    # Pretrain model with SimCLR
    print("Pretraining model with SimCLR...")
    simclr_trainer.train(
        train_loader=simclr_loader, 
        epochs=100
    )

    # Fine-tune model on labeled MNIST data
    print("Fine-tuning model on labeled MNIST data...")
    simclr_trainer.fine_tune(
        train_loader=finetune_loader,
        test_loader=test_loader,
        epochs=10,
        evaluate_every=1,
        lr = 0.002
    )

    # Initialize a second model (no SimCLR pretraining)
    model_supervised = CNN(use_dropout=True, dropout_rate=0.3, use_projection_head=True)

    # Use the same trainer class for convenience
    supervised_trainer = SimCLRTrainer(model_supervised)

    # Directly fine-tune on the finetune_loader
    print("Fine-tuning model without SimCLR pretraining...")
    supervised_trainer.fine_tune(
        train_loader=finetune_loader,
        test_loader=test_loader,
        epochs=10,
        evaluate_every=1
    )

if __name__ == '__main__':
    main()