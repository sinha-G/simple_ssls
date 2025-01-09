# trainers/dino_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import ImageOps
from datetime import datetime
from pathlib import Path
from .semi_supervised_base import SemiSupervisedTrainer
from copy import deepcopy

# Define DINO augmentations
class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

class TensorSolarization:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def __call__(self, x):
        return torch.where(x >= self.threshold, 1-x, x)

# Global views transformations for tensors
global_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    lambda x: x + torch.randn_like(x) * 0.1 if torch.rand(1) < 0.5 else x,  # Gaussian noise
    TensorSolarization(threshold=0.5),
])

# Local views transformations for tensors
local_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 0.4)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    lambda x: x + torch.randn_like(x) * 0.1 if torch.rand(1) < 0.5 else x,  # Gaussian noise
])

class DINOTrainer(torch.nn.Module, SemiSupervisedTrainer):
    def __init__(self, model, 
                 lr=0.0001,
                 momentum_teacher=0.996,
                 momentum_center=0.9,
                 temp_student=0.1,
                 temp_teacher=0.04,
                 n_local_views=6,
                 weight_decay=0.04):
        super().__init__()
        SemiSupervisedTrainer.__init__(self, model)
        
        # Model setup with proper initialization
        self.student = model.to(self.device)
        self.teacher = deepcopy(model).to(self.device)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,
            eta_min=1e-6
        )
        
        self.momentum_teacher = momentum_teacher
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        
        # Initialize center for loss computation
        self.register_buffer("center", torch.zeros(1, model.num_classes))

    def train_step(self, images):
        images = images.to(self.device)
        
        # Check for NaN inputs
        if torch.isnan(images).any():
            print("NaN detected in input images")
            return None
            
        # Forward passes
        with torch.no_grad():
            teacher_output = self.teacher(images)[0]  # Get logits output
        student_outputs = self.student(images)
        student_output = student_outputs[0]  # Get logits output
        
        # Check outputs before loss
        if torch.isnan(student_output).any():
            print(f"NaN in student output. Max: {student_output.max()}, Min: {student_output.min()}")
            return None
            
        loss = self.dino_loss(teacher_output, student_output)
        
        if torch.isnan(loss):
            print("NaN loss detected")
            return None
        
        return loss

    def train(self, train_loader, test_loader, epochs=100, evaluate_every=5):
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            valid_batches = 0
            
            for batch_idx, (images, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                loss = self.train_step(images)
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                    self.update_teacher()
                    total_loss += loss.item()
                    valid_batches += 1
                
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                self.scheduler.step()
            
            # Early stopping on NaN
            if valid_batches == 0:
                print("Training failed - too many NaN values")
                break

    @torch.no_grad()
    def update_teacher(self):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = self.momentum_teacher * param_t.data + (1 - self.momentum_teacher) * param_s.data
            
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0)
        if self.center is None:
            self.center = batch_center
        else:
            self.center = self.momentum_center * self.center + (1 - self.momentum_center) * batch_center

    def dino_loss(self, teacher_output, student_output):
        print(f"\nDebug Loss Computation:")
        print(f"\nStudent output magnitude: {torch.norm(student_output)}")
        print(f"Student output max/min: {student_output.max()}/{student_output.min()}")
        print(f"Teacher output shape: {teacher_output.shape}")
        print(f"Student output shape: {student_output.shape}")
        print(f"Contains NaN - Teacher: {torch.isnan(teacher_output).any()}, Student: {torch.isnan(student_output).any()}")
        if torch.norm(student_output) > 1e3:
            print("WARNING: Large student output detected!")

        teacher_output = teacher_output.detach()

        # Clip extreme values
        student_output = torch.clamp(student_output, -100, 100)
        teacher_output = torch.clamp(teacher_output, -100, 100)
        
        # Normalize outputs
        teacher_output = F.normalize(teacher_output, dim=-1, p=2)
        student_output = F.normalize(student_output, dim=-1, p=2)
        print(f"After norm - Contains NaN - Teacher: {torch.isnan(teacher_output).any()}, Student: {torch.isnan(student_output).any()}")
        
        # Temperature scaling
        teacher_out = F.softmax(teacher_output/self.temp_teacher, dim=-1)
        student_out = F.softmax(student_output/self.temp_student, dim=-1)
        print(f"After softmax - Contains NaN - Teacher: {torch.isnan(teacher_out).any()}, Student: {torch.isnan(student_out).any()}")
        
        eps = 1e-7
        student_out = student_out + eps
        
        loss = -torch.mean(torch.sum(teacher_out * torch.log(student_out), dim=-1))
        print(f"Final loss: {loss.item()}")
        
        if torch.isnan(loss):
            print("WARNING: Loss is NaN!")
            print(f"Teacher range: [{teacher_out.min()}, {teacher_out.max()}]")
            print(f"Student range: [{student_out.min()}, {student_out.max()}]")
        
        return loss

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.student.eval()
        correct = 0
        total = 0
        class_correct = [0 for _ in range(10)]
        class_total = [0 for _ in range(10)]
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Modified line: Only unpack the first return value
            logits, *_ = self.student(images)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        
        accuracy = 100 * correct / total
        per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
        
        self.student.train()
        return accuracy, class_correct, class_total