import pytorch_lightning as pl
from argparse import Namespace
import typing as T

from torch.optim import lr_scheduler
from model.net import AMLSimNet, CitationNet
import torch_geometric as pyg
import torch.nn.functional as F
import torch
from dataset import AMLSimDataset, BatchAMLSimDataset
from transforms.normalize import NormalizeFeatures
from transforms.graph import AttachEdgeAttr, AddSelfLoop
from torch_geometric.transforms import Compose
from transforms.dataset import AttachEdgeLabel


class EGAT_trainer(pl.LightningModule):
    def __init__(self, hparams: T.Union[T.Dict, Namespace]):
        super(EGAT_trainer, self).__init__()
        if isinstance(hparams, Namespace):
            self.hparams = hparams
        else:
            self.hparams = Namespace(**hparams)
        dataset = self.dataset()
        if isinstance(hparams, Namespace):
            hparams = vars(hparams)
        self.model = AMLSimNet(
            vertex_in_feature=dataset.num_node_features,
            edge_in_feature=dataset.num_edge_features,
            num_classes=dataset.num_classes,
            **hparams
        )
        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            self.weight = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1]).cuda()
        else:
            self.weight = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1]).cuda()
        self.category = ["scatter_gather", "gather_scatter", "cycle", "fan_in", "fan_out", "bipartite", "stack"]

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data: pyg.data.Data = batch

        y_pred, edge_pred = self.model(data)
        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            y_pred_train = y_pred
            y_true_train = data.y
            edge_pred_train = edge_pred
            edge_true_train = data.edge_label
        else:
            y_pred_train = y_pred[data.train_mask]
            y_true_train = data.y[data.train_mask]
            edge_pred_train = edge_pred
            edge_true_train = data.edge_label

        y_true_train = y_true_train.to(y_pred.device)
        edge_true_train = edge_true_train.to(y_pred.device)

        loss = F.cross_entropy(y_pred_train, y_true_train, weight=self.weight) + F.cross_entropy(edge_pred_train, edge_true_train)

        tensorboard_log = {
            "train_loss": loss
        }

        return {"loss": loss, "log": tensorboard_log}

    def validation_step(self, batch, batch_idx):
        data: pyg.data.Data = batch

        y_pred, _ = self.model(data)

        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            y_pred_val = y_pred
            y_true_val = data.y
        else:
            y_pred_val = y_pred[data.val_mask]
            y_true_val = data.y[data.val_mask]

        y_true_val = y_true_val.to(y_pred.device)

        loss = F.cross_entropy(y_pred_val, y_true_val, weight=self.weight)

        choose = torch.argmax(y_pred_val, dim=-1)
        correct = (choose == y_true_val)
        total = y_pred_val.size(0)

        category_recall_correct = []
        category_recall_total = []
        for label in range(1, y_true_val.max() + 1):
            category_recall_correct.append(correct[y_true_val == label].sum())
            category_recall_total.append((y_true_val == label).sum())

        category_precise_correct = []
        category_precise_total = []
        for label in range(1, y_true_val.max() + 1):
            category_precise_correct.append(correct[choose == label].sum())
            category_precise_total.append((choose == label).sum())

        return {
            "val_loss": loss,
            "correct": correct.sum(),
            "total": total,
            "category_recall_correct": category_recall_correct,
            "category_recall_total": category_recall_total,
            "category_precise_correct": category_precise_correct,
            "category_precise_total": category_precise_total
        }

    def test_step(self, batch, batch_idx):
        data: pyg.data.Data = batch

        y_pred, _ = self.model(data)

        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            y_pred_test = y_pred
            y_true_test = data.y
        else:
            y_pred_test = y_pred[data.test_mask]
            y_true_test = data.y[data.test_mask]

        y_true_test = y_true_test.to(y_pred.device)

        loss = F.cross_entropy(y_pred_test, y_true_test, weight=self.weight)

        choose = torch.argmax(y_pred_test, dim=-1)
        correct = (choose == y_true_test)
        total = y_pred_test.size(0)

        category_recall_correct = []
        category_recall_total = []
        for label in range(1, y_true_test.max() + 1):
            category_recall_correct.append(correct[y_true_test == label].sum())
            category_recall_total.append((y_true_test == label).sum())

        category_precise_correct = []
        category_precise_total = []
        for label in range(1, y_true_test.max() + 1):
            category_precise_correct.append(correct[choose == label].sum())
            category_precise_total.append((choose == label).sum())

        return {
            "test_loss": loss,
            "correct": correct.sum(),
            "total": total,
            "category_recall_correct": category_recall_correct,
            "category_recall_total": category_recall_total,
            "category_precise_correct": category_precise_correct,
            "category_precise_total": category_precise_total,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output["val_loss"]
                                for output in outputs]).mean()
        correct = torch.stack([output["correct"] for output in outputs]).sum()
        total = sum([output["total"] for output in outputs])
        acc = correct.float() / total

        category_recall_correct = torch.tensor(
            [output["category_recall_correct"] for output in outputs]).sum(dim=0).float()
        category_recall_total = torch.tensor(
            [output["category_recall_total"] for output in outputs]).sum(dim=0).float()

        category_recall = category_recall_correct / (category_recall_total + torch.ones_like(category_recall_correct) * 0.0001)

        category_precise_correct = torch.tensor(
            [output["category_precise_correct"] for output in outputs]).sum(dim=0).float()
        category_precise_total = torch.tensor(
            [output["category_precise_total"] for output in outputs]).sum(dim=0).float()

        category_precise = category_precise_correct / (category_precise_total + torch.ones_like(category_precise_correct) * 0.0001)

        category_f1 = 2 * (category_precise * category_recall) / (category_precise + category_recall + torch.ones_like(category_precise) * 0.0001)
        mean_f1 = category_f1.mean()

        tensorboard_log = {
            "val_loss": avg_loss,
            "val_acc": acc,
            "val_f1": mean_f1
        }

        return {"val_loss": avg_loss, "val_acc": acc, "val_f1": mean_f1, "log": tensorboard_log, 'progress_bar': {"val_loss": avg_loss, "val_f1": mean_f1}}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output["test_loss"]
                                for output in outputs]).mean()
        correct = torch.stack([output["correct"] for output in outputs]).sum()
        total = sum([output["total"] for output in outputs])
        acc = correct.float() / total

        category_recall_correct = torch.tensor(
            [output["category_recall_correct"] for output in outputs]).sum(dim=0).float()
        category_recall_total = torch.tensor(
            [output["category_recall_total"] for output in outputs]).sum(dim=0).float()

        category_recall = category_recall_correct / (category_recall_total + torch.ones_like(category_recall_correct) * 0.0001)

        category_precise_correct = torch.tensor(
            [output["category_precise_correct"] for output in outputs]).sum(dim=0).float()
        category_precise_total = torch.tensor(
            [output["category_precise_total"] for output in outputs]).sum(dim=0).float()

        category_precise = category_precise_correct / (category_precise_total + torch.ones_like(category_recall_correct) * 0.0001)

        category_f1 = 2 * (category_precise * category_recall) / (category_precise + category_recall + torch.ones_like(category_precise) * 0.0001)
        mean_f1 = category_f1.mean()

        tensorboard_log = {
            "test_loss": avg_loss,
            "test_acc": acc,
            "test_f1": mean_f1
        }

        f1_dict = dict([(self.category[i], f1) for (i, f1) in enumerate(category_f1)])

        return {"test_loss": avg_loss, "test_acc": acc, "log": tensorboard_log, "test_f1": mean_f1, **f1_dict}

    def configure_optimizers(self):
        op = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2norm)
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode='min',
                                                              factor=0.5, patience=50,
                                                              min_lr=0.00001)
        return {
            "optimizer": op,
            "lr_scheduler": schedule,
            "monitor": "val_loss"
        }

    def dataset(self):
        if self.hparams.dataset == "AMLSim-1K-merge":
            return AMLSimDataset(root="data/AMLSim/1K-merge", mode=self.hparams.mode, transform=NormalizeFeatures())
        elif self.hparams.dataset == "AMLSim-10K-merge":
            return AMLSimDataset(root="data/AMLSim/10K-merge", mode=self.hparams.mode, transform=NormalizeFeatures())
        elif self.hparams.dataset == "AMLSim-10K-merge-hard":
            return AMLSimDataset(root="data/AMLSim/10K-merge-hard", mode=self.hparams.mode, transform=NormalizeFeatures())
        elif self.hparams.dataset == "AMLSim-100K-merge":
            return AMLSimDataset(root="data/AMLSim/100K-merge", mode=self.hparams.mode, transform=NormalizeFeatures())
        elif self.hparams.dataset == "AMLSim-100K-merge-hard":
            return AMLSimDataset(root="data/AMLSim/100K-merge-hard", mode=self.hparams.mode, transform=NormalizeFeatures())
        elif self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            return BatchAMLSimDataset(root="data/AMLSim/10K-merge-hard-batch", transform=Compose([
                # AddSelfLoop("zero"),
                NormalizeFeatures(),
                AttachEdgeLabel()
            ]))
        else:
            raise ValueError("Unknown dataset")

    def train_dataloader(self):
        dataset = self.dataset()
        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            return pyg.data.DataLoader(dataset[: 5], num_workers=0)
        else:
            return pyg.data.DataLoader(dataset, num_workers=0)

    def val_dataloader(self):
        dataset = self.dataset()
        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            return pyg.data.DataLoader(dataset[5: 10], num_workers=0)
        else:
            return pyg.data.DataLoader(dataset, num_workers=0)

    def test_dataloader(self):
        dataset = self.dataset()
        if self.hparams.dataset == "AMLSim-10K-merge-hard-batch":
            return pyg.data.DataLoader(dataset[10:], num_workers=0)
        else:
            return pyg.data.DataLoader(dataset, num_workers=0)
