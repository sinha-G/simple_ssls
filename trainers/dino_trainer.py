import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

from torchvision import transforms
from datetime import datetime
from .semi_supervised_base import SemiSupervisedTrainer
from copy import deepcopy
from tqdm.auto import tqdm


class TensorSolarization:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def __call__(self, x):
        return torch.where(x >= self.threshold, 1-x, x)

class DINOTrainer(torch.nn.Module, SemiSupervisedTrainer):
    def __init__(self, model, 
                 lr=0.0001,
                 momentum_teacher=0.996,
                 momentum_center=0.9,
                 temp_student=0.1,
                 temp_teacher=0.04,
                 n_local_views=6,
                 weight_decay=0.04,
                 patience=10):
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
        self.patience = patience
        self.best_loss = float('inf')

        # Initialize center for loss computation
        self.register_buffer("center", torch.zeros(1, model.num_classes))

    def train_step(self, images):
        images = images.to(self.device)
        
        # Check for NaN inputs
        # if torch.isnan(images).any():
        #     print("NaN detected in input images")
        #     return None
            
        # Forward passes
        with torch.no_grad():
            teacher_output = self.teacher(images)[0]  # Get logits output
        student_outputs = self.student(images)
        student_output = student_outputs[0]  # Get logits output
        
        # Check outputs before loss
        # if torch.isnan(student_output).any():
        #     print(f"NaN in student output. Max: {student_output.max()}, Min: {student_output.min()}")
        #     return None
            
        loss = self.dino_loss(teacher_output, student_output)
        
        # if torch.isnan(loss):
        #     print("NaN loss detected")
        #     return None
        
        return loss

    def train(self, train_loader, test_loader, epochs=100):
        history = {
            'train_loss': [],
            'lr': []
        }
        patience_counter = 0
        best_model_state = None
        max_grad_norm = 1.0
        
        # try:
        for epoch in tqdm(range(epochs), desc='Epochs', position=0):
            self.student.train()
            total_loss = 0
            valid_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', position=1, leave=False)
            for batch_idx, (images, _) in enumerate(pbar):
                self.optimizer.zero_grad()
                # try:
                loss = self.train_step(images)
            
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), 
                        max_grad_norm
                    )
                    self.optimizer.step()
                    self.update_teacher()
                    # torch.cuda.synchronize()
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                history['train_loss'].append(avg_loss)
                history['lr'].append(self.scheduler.get_last_lr()[0])
                
                # Early stopping
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = self.student.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} Epoch [{epoch+1}/{epochs}], "
                    f"Loss: {avg_loss:.6f}, "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}, "
                    f"Patience: {patience_counter}/{self.patience}")

                self.scheduler.step()
                
            if valid_batches == 0:
                print("Training failed - too many NaN values")
                break
        
        if best_model_state is not None:
            self.student.load_state_dict(best_model_state)

        # except Exception as e:
            # print(f"Training error: {str(e)}")
            # if best_model_state is not None:
            #     self.student.load_state_dict(best_model_state)

        return history

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
        # print(f"\nDebug Loss Computation:")
        # print(f"\nStudent output magnitude: {torch.norm(student_output)}")
        # print(f"Student output max/min: {student_output.max()}/{student_output.min()}")
        # print(f"Teacher output shape: {teacher_output.shape}")
        # print(f"Student output shape: {student_output.shape}")
        # print(f"Contains NaN - Teacher: {torch.isnan(teacher_output).any()}, Student: {torch.isnan(student_output).any()}")
        # if torch.norm(student_output) > 1e3:
        #     print("WARNING: Large student output detected!")

        teacher_output = teacher_output.detach()

        # Clip extreme values
        student_output = torch.clamp(student_output, -100, 100)
        teacher_output = torch.clamp(teacher_output, -100, 100)
        
        # Normalize outputs
        teacher_output = F.normalize(teacher_output, dim=-1, p=2)
        student_output = F.normalize(student_output, dim=-1, p=2)
        # print(f"After norm - Contains NaN - Teacher: {torch.isnan(teacher_output).any()}, Student: {torch.isnan(student_output).any()}")
        
        # Temperature scaling
        teacher_out = F.softmax(teacher_output/self.temp_teacher, dim=-1)
        student_out = F.softmax(student_output/self.temp_student, dim=-1)
        # print(f"After softmax - Contains NaN - Teacher: {torch.isnan(teacher_out).any()}, Student: {torch.isnan(student_out).any()}")
        
        eps = 1e-7
        student_out = student_out + eps
        
        loss = -torch.mean(torch.sum(teacher_out * torch.log(student_out), dim=-1))
        # print(f"Final loss: {loss.item()}")
        
        # if torch.isnan(loss):
        #     print("WARNING: Loss is NaN!")
        #     print(f"Teacher range: [{teacher_out.min()}, {teacher_out.max()}]")
        #     print(f"Student range: [{student_out.min()}, {student_out.max()}]")
        
        return loss

    def finetune(self, train_loader, test_loader, epochs=100, lr=0.001, patience=10, evaluate_every=1, visualize_every=1, 
                 checkpoint_path='checkpoints', resume_from=None):
        
        # Setup for finetuning
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.student.train()
        
        # Group parameters for weight decay
        decay = []
        no_decay = []
        
        for name, param in self.student.named_parameters():
            if 'bias' in name or 'norm' in name or 'pos_embed' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        # Configure optimizer with weight decay
        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': 0.03},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience ** 0.5)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        start_epoch = 0
        best_acc = 0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'lr': []
        }
        
        if resume_from and os.path.exists(resume_from):
            print(f"Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from)
            self.student.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            history = checkpoint['history']
            print(f"Resuming from epoch {start_epoch} with best accuracy {best_acc:.2f}%")

        # Add epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc='Epochs', position=0)

        # with open('data/imagenet/imagenet1000_clsidx_to_labels.txt', 'r') as f:
        #     class_labels = eval(f.read())

        for epoch in epoch_pbar:
            torch.cuda.empty_cache()

            # Training
            self.student.train()
            scaler = torch.amp.GradScaler('cuda')
            total_loss = 0
            correct = 0
            total = 0
            
            # Add batch progress bar
            batch_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')#,leave=False, position=1)
            
            for images, labels in batch_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits = self.student(images)[0]  # Get classification output
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Evaluate
            if (epoch + 1) % evaluate_every == 0:
                test_acc, _, _, val_loss = self.evaluate(test_loader)
                scheduler.step(val_loss)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model_state = self.student.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("\nEarly stopping triggered")
                        break
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Train Acc': f'{train_acc:.2f}%',
                    'Test Acc': f'{test_acc:.2f}%'
                })

            # Visualize attention every `visualize_every` epochs, if set
            if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
                images, _ = next(iter(test_loader))
                images = images.to(self.device)
                random_idx = random.sample(range(images.size(0)), k=2)
                

                print(f"\nVisualizing attention at epoch {epoch+1}...")
                for i in random_idx:  # Visualize a couple of images
                    for layer in range(self.student.n_blocks):
                        fig, axes = plt.subplots(1, min(self.student.n_heads, 4), figsize=(20, 5))
                        if self.student.n_heads == 1:
                            axes = [axes]
                        for head in range(min(self.student.n_heads, 4)):  # Show up to 4 heads
                            self.student.visualize_attention(
                                images=test_loader.dataset[i][0].unsqueeze(0).to('cuda'),
                                layer_idx=layer,
                                head_idx=head,
                                alpha=0.6,
                                ax=axes[head]
                            )
                        class_idx = test_loader.dataset[i][1]
                        # class_name = class_labels[class_idx]
                        plt.suptitle(f'Layer {layer} Attention Maps\nClass: {class_idx}', size=16)
                        plt.tight_layout()
                        plt.show()
            
            # Update history
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc if 'test_acc' in locals() else 0)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
            # After evaluation, save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(
                checkpoint_path, 
                f'checkpoint_epoch_{epoch+1}_{timestamp}.pt'
            )
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state': self.student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }
            
            # Save as latest checkpoint and best checkpoint if applicable
            torch.save(checkpoint, os.path.join(checkpoint_path, 'latest_checkpoint.pt'))
            if 'test_acc' in locals() and test_acc >= best_acc:
                torch.save(checkpoint, os.path.join(checkpoint_path, 'best_checkpoint.pt'))
                
            print(f"Checkpoint saved to {checkpoint_file}")
            
            # Early stopping check
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
        
        return history

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.student.eval()
        correct = 0
        total = 0

        num_classes = self.student.num_classes
        class_correct = [0 for _ in range(num_classes)]
        class_total = [0 for _ in range(num_classes)]
        val_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Modified line: Only unpack the first return value
            logits, *_ = self.student(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                if 0 <= label < num_classes:  # Validate label is in range
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        accuracy = 100 * correct / total
        per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
        avg_val_loss = val_loss / len(test_loader)

        self.student.train()
        return accuracy, class_correct, class_total, avg_val_loss