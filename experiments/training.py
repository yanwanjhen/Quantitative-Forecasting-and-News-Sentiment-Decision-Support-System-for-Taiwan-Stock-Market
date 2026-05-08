"""
模型訓練模組（回歸版）
Model Training Module
"""

import copy
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def resolve_device(config, device: str = None):
    if device is not None:
        return torch.device(device)

    if getattr(config, "use_gpu", False):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-6,
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        cmp_score = -score if self.mode == 'min' else score

        if self.best_score is None:
            self.best_score = cmp_score
            self.best_state = copy.deepcopy(model.state_dict())
        elif cmp_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = cmp_score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class HybridFinancialLoss(nn.Module):
    """
    比較接近你先前 >60% 路線的回歸混合損失
    """
    def __init__(self, alpha_reg=0.8, alpha_dir=0.8, beta=1.0, alpha_var=0.02):
        super().__init__()
        self.alpha_reg = alpha_reg
        self.alpha_dir = alpha_dir
        self.alpha_var = alpha_var
        self.reg_loss = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reg = self.reg_loss(pred, target)

        mag_weight = 1.0 + 0.3 * torch.abs(target)
        mag_weight = torch.clamp(mag_weight, min=1.0, max=4.0)
        reg = reg * mag_weight

        target_sign = torch.sign(torch.where(target == 0, torch.ones_like(target), target))
        dir_margin = pred * target_sign
        dir_loss = torch.log1p(torch.exp(-3.0 * dir_margin)) / 3.0
        dir_loss = dir_loss * mag_weight

        base = self.alpha_reg * reg + self.alpha_dir * dir_loss

        if pred.numel() > 1:
            var_bonus = self.alpha_var * torch.var(pred, unbiased=False)
            base = base + var_bonus

        return base


class Trainer:
    def __init__(self, model: nn.Module, config, device: str = None,
                 model_type: str = 'single', use_weighted_loss: bool = False):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.use_weighted_loss = use_weighted_loss

        self.device = resolve_device(config, device)
        self.model.to(self.device)

        self.criterion = HybridFinancialLoss(alpha_reg=0.55, alpha_dir=1.20, beta=1.0, alpha_var=0.04)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.early_stopping = EarlyStopping(
            patience=config.model.patience,
            mode='min',
            verbose=config.verbose
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }

    def compute_class_weights(self, regime_labels: np.ndarray) -> torch.Tensor:
        regime_labels = np.array(regime_labels)
        n_bull = np.sum(regime_labels == 0)
        n_bear = np.sum(regime_labels == 1)
        total = len(regime_labels)

        if n_bear == 0 or n_bull == 0:
            return torch.ones(total, device=self.device)

        weight_bull = total / (2 * n_bull)
        weight_bear = total / (2 * n_bear)
        weights = np.where(regime_labels == 0, weight_bull, weight_bear)

        print(f"  📊 樣本權重: 牛市(n={n_bull})={weight_bull:.3f}, 熊市(n={n_bear})={weight_bear:.3f}")
        return torch.FloatTensor(weights).to(self.device)

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        X = batch['X'].to(self.device)
        y = batch['y'].to(self.device)

        if self.model_type in ['regime_embedding', 'regime_switch']:
            regime = batch['regime'].to(self.device)
            pred = self.model(X, regime)
        else:
            pred = self.model(X)

        return pred, y

    def train_epoch(self, train_loader: DataLoader, sample_weights: torch.Tensor = None) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0
        batch_start_idx = 0

        for batch in train_loader:
            pred, y = self._forward_batch(batch)
            batch_size = y.size(0)

            self.optimizer.zero_grad()
            losses = self.criterion(pred, y)

            if self.use_weighted_loss and sample_weights is not None:
                batch_weights = sample_weights[batch_start_idx:batch_start_idx + batch_size]
                if len(batch_weights) == batch_size:
                    loss = (losses * batch_weights).mean()
                else:
                    loss = losses.mean()
            else:
                loss = losses.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_mae += torch.abs(pred - y).sum().item()
            n_samples += batch_size
            batch_start_idx += batch_size

        return total_loss / max(n_samples, 1), total_mae / max(n_samples, 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                pred, y = self._forward_batch(batch)
                batch_size = y.size(0)

                loss = self.criterion(pred, y).mean()

                total_loss += loss.item() * batch_size
                total_mae += torch.abs(pred - y).sum().item()
                n_samples += batch_size

        return total_loss / max(n_samples, 1), total_mae / max(n_samples, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = None, verbose: bool = True,
              train_regime_labels: np.ndarray = None) -> Dict:
        if epochs is None:
            epochs = self.config.model.epochs

        print(f"🚀 開始訓練 ({self.device})...")
        print(f"   模型類型: {self.model_type}")
        print(f"   訓練輪數: {epochs}")
        print(f"   加權損失: {'✅ 啟用' if self.use_weighted_loss else '❌ 停用'}")

        sample_weights = None
        if self.use_weighted_loss and train_regime_labels is not None:
            sample_weights = self.compute_class_weights(train_regime_labels)

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_mae = self.train_epoch(train_loader, sample_weights)
            val_loss, val_mae = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                      f"Train MAE={train_mae:.6f}, Val MAE={val_mae:.6f}, "
                      f"Time={epoch_time:.1f}s")

            if self.early_stopping(val_loss, self.model):
                print(f"  ⚠️ 早停 at epoch {epoch+1}")
                break

        self.early_stopping.load_best_model(self.model)

        total_time = time.time() - start_time
        print(f"✅ 訓練完成! 總時間: {total_time/60:.1f} 分鐘")
        best_val = -self.early_stopping.best_score if self.early_stopping.best_score is not None else float('nan')
        print(f"   最佳驗證損失: {best_val:.6f}")

        return self.history

    def predict(self, X: np.ndarray, regime: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if self.model_type in ['regime_embedding'] and regime is not None:
                regime_tensor = torch.LongTensor(regime).to(self.device)
                pred = self.model(X_tensor, regime_tensor)
            else:
                pred = self.model(X_tensor)

        return pred.cpu().numpy()

    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in test_loader:
                pred, y = self._forward_batch(batch)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
        if np.isnan(correlation):
            correlation = 0.0
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'correlation': correlation,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': self.model_type,
            'history': self.history
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        return checkpoint


class RegimeSwitchTrainer:
    """
    RegimeSwitch 專用訓練器（做法 A）
    """

    def __init__(self, model: nn.Module, config, device: str = None):
        self.model = model
        self.config = config
        self.device = resolve_device(config, device)
        self.model.to(self.device)

        self.reg_criterion = HybridFinancialLoss(alpha_reg=0.55, alpha_dir=1.20, beta=1.0, alpha_var=0.04)
        self.dir_criterion = nn.BCEWithLogitsLoss()
        self.regime_criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.history = {'train_loss': [], 'val_loss': []}

    def _make_loader(self, split: Dict, batch_size: int):
        from data_splitting import SequenceDataset
        dataset = SequenceDataset(split['X'], split['y'], split['regime'], split['dates'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            regime = batch['regime'].to(self.device).long()

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                out = self.model(X, regime_target=regime, return_aux=True)

                reg_pred = out['regression']
                direction_logits = out['direction_logits']
                regime_logits = out['regime_logits']

                reg_loss = self.reg_criterion(reg_pred, y).mean()
                direction_target = (y > 0).float()
                dir_loss = self.dir_criterion(direction_logits, direction_target)
                regime_loss = self.regime_criterion(regime_logits, regime)

                loss = reg_loss + 0.20 * dir_loss + 0.30 * regime_loss

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, data: Dict, epochs: int = None) -> Dict:
        if epochs is None:
            epochs = self.config.model.epochs

        batch_size = self.config.model.batch_size
        train_loader = self._make_loader(data['train'], batch_size)
        val_loader = self._make_loader(data['val'], batch_size)

        early_stopping = EarlyStopping(
            patience=self.config.model.patience,
            mode='min',
            verbose=self.config.verbose
        )

        print(f"🚀 開始訓練 ({self.device})...")
        print("   模型類型: regime_switch")
        print(f"   訓練輪數: {epochs}")

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")

            if early_stopping(val_loss, self.model):
                print(f"  ⚠️ 早停 at epoch {epoch+1}")
                break

        early_stopping.load_best_model(self.model)
        return self.history

    def predict(self, X: np.ndarray):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            out = self.model(X_tensor, regime_target=None, return_aux=True)

        predictions = out['regression'].cpu().numpy()
        predicted_regime = out['predicted_regime'].cpu().numpy()

        return predictions, predicted_regime


if __name__ == "__main__":
    print("訓練模組載入成功!")
