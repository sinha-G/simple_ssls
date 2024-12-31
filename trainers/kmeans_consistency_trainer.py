# trainers/kmeans_consistency_trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from torchvision import transforms
from .semi_supervised_base import SemiSupervisedTrainer

# Define transformations for consistency regularization
consistency_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.85, 1.15)),
    transforms.Normalize((0.5,), (0.5,))
])

class KMeansConsistencyTrainer(SemiSupervisedTrainer):
    def __init__(self, model, device='cuda', lambda_kmeans=0.002, lambda_consistency=0.0002, use_consistency=True, use_unlabeled=True):
        super().__init__(model, device)
        self.lambda_kmeans = lambda_kmeans
        self.lambda_consistency = lambda_consistency
        self.use_consistency = use_consistency
        self.use_unlabeled = use_unlabeled

    def train(self, labeled_loader, unlabeled_loader, test_loader, epochs=10, evaluate_every=1):
        train_accs = []
        test_accs = []

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for (x_labeled, y_labeled) in labeled_loader:
                x_labeled, y_labeled = x_labeled.to(self.device), y_labeled.to(self.device)
                optimizer.zero_grad()

                logits, z_labeled = self.model(x_labeled)
                loss_sup = criterion(logits, y_labeled)
                total_loss += loss_sup.item()

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y_labeled).sum().item()
                total += y_labeled.size(0)

                if self.use_unlabeled:
                    x_unlabeled, _ = next(iter(unlabeled_loader))
                    x_unlabeled = x_unlabeled.to(self.device)
                    _, z_unlabeled = self.model(x_unlabeled)

                    kmeans = KMeans(n_clusters=10, random_state=0).fit(z_unlabeled.detach().cpu().numpy())
                    cluster_centers = torch.tensor(kmeans.cluster_centers_, device=z_unlabeled.device)
                    loss_kmeans = torch.mean(torch.min(torch.cdist(z_unlabeled, cluster_centers), dim=1)[0])

                    if self.use_consistency:
                        x_unlabeled_aug = torch.stack([consistency_transform(img) for img in x_unlabeled]).to(self.device)
                        _, z_unlabeled_aug = self.model(x_unlabeled_aug)
                        loss_consistency = torch.mean((z_unlabeled - z_unlabeled_aug) ** 2)
                    else:
                        loss_consistency = 0

                    loss = loss_sup + self.lambda_kmeans * loss_kmeans + self.lambda_consistency * loss_consistency
                else:
                    loss = loss_sup

                loss.backward()
                optimizer.step()

            scheduler.step()

            train_acc = 100 * correct / total
            train_accs.append(train_acc)

            if (epoch + 1) % evaluate_every == 0 or epoch == 0:
                test_accuracy, _, _ = self.evaluate(test_loader)
                test_accs.append(test_accuracy)
            else:
                test_accs.append(test_accs[-1])

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(labeled_loader)}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_accs[-1]:.2f}%")

        return train_accs, test_accs

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = self.model(images)
                predictions = torch.argmax(logits, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy, np.array(all_preds), np.array(all_labels)