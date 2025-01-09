# trainers/simclr_trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
from pathlib import Path
from .semi_supervised_base import SemiSupervisedTrainer

# Define SimCLR augmentations
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale = (0.5, 1.0)),
    transforms.RandomRotation(degrees=45),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class SimCLRTrainer:
    def __init__(self, model, lr_simclr = 0.003, lr_finetune = 0.002, optimizer = None, temperature=0.5, checkpoint_dir='checkpoints'):
        self.model = model
        self.lr_simclr = lr_simclr
        self.lr_finetune = lr_finetune
        self.optimizer_simclr = optim.Adam(self.model.parameters(), lr=self.lr_simclr, weight_decay=1e-5) if optimizer is None else optimizer
        self.optimizer_finetune = optim.Adam(self.model.parameters(), lr=self.lr_finetune)
        self.temperature = temperature
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'temperature': self.temperature,
            'best_loss': self.best_loss
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return False
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.temperature = checkpoint['temperature']
        self.best_loss = checkpoint['best_loss']
        print(f"Resumed from epoch {checkpoint['epoch']}")
        return True
    
    def contrastive_loss(self, projections):
        batch_size = projections.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        projections = F.normalize(projections, dim=1)
        similarity_matrix = torch.matmul(projections, projections.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature
        return F.cross_entropy(logits, labels)

    def train(self, train_loader, epochs=10):
        # train_losses = []
        train_accs = []
        test_accs = []
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_simclr, 
            T_max=epochs
        )

        # print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            # correct = 0
            # total = 0
            
            for x, _ in train_loader:
                # Create two augmented views
                x = torch.cat([simclr_transform(img).unsqueeze(0) for img in x], dim=0)
                x = torch.cat([x, x], dim=0)
                x = x.to(self.device)

                self.optimizer_simclr.zero_grad()
                logits, _, projections = self.model(x)
                loss = self.contrastive_loss(projections)
                loss.backward()
                self.optimizer_simclr.step()

                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"[{timestamp}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
                
                self.save_checkpoint(
                    epoch=epoch,
                    loss=avg_loss,
                    is_best=is_best
                )

        return train_accs, test_accs

    def fine_tune(self, train_loader, test_loader, lr=0.001, epochs=25, evaluate_every=1, patience=20, scheduler_patience=10):
        train_accs = []
        test_accs = []
        best_loss = float('inf')
        best_acc = 0
        patience_counter = 0
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_finetune,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            verbose=True
        )
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer_finetune.zero_grad()
                logits, _, _ = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                self.optimizer_finetune.step()

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss/len(train_loader)
            train_acc = 100 * correct / total
            train_accs.append(train_acc)
            
            # Evaluate and check for early stopping
            if (epoch + 1) % evaluate_every == 0 or epoch == 0:
                test_accuracy, _, _, val_loss = self.evaluate(test_loader)
                test_accs.append(test_accuracy)
                
                scheduler.step(val_loss)

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    best_model_state = self.model.state_dict()
                    
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
            else:
                test_accs.append(test_accs[-1])

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} Epoch [{epoch+1}/{epochs}], "
                f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                f"Loss: {avg_loss:.4f}, "
                f"Train Accuracy: {train_acc:.2f}%, "
                f"Test Loss: {val_loss:.4f}, "
                f"Test Accuracy: {test_accs[-1]:.2f}%")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return train_accs, test_accs, best_acc

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _, _ = self.model(images)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(data_loader)
        return accuracy, all_preds, all_labels, avg_val_loss
