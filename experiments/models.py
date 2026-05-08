"""
LSTM 模型模組
LSTM Model Module for Time Series Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class LSTMEncoder(nn.Module):
    """LSTM 編碼器"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
        """
        初始化 LSTM 編碼器
        
        Args:
            input_size: 輸入特徵維度
            hidden_size: 隱藏層維度
            num_layers: LSTM 層數
            dropout: Dropout 比率
            bidirectional: 是否使用雙向 LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            output: (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): 最終隱藏狀態
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class AttentionLayer(nn.Module):
    """注意力層 - 用於 LSTM 輸出加權"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            # 移除過度壓縮，確保特徵完整性
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            
        Returns:
            context: (batch, hidden_size) - 加權後的上下文向量
        """
        # 計算注意力權重
        attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加權求和
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context


class TimeSeriesLSTM(nn.Module):
    """
    時間序列 LSTM 模型 (基礎版本)
    """
    
    def __init__(self, n_features: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2,
                 seq_len: int = 60, use_attention: bool = True):
        """
        初始化模型
        
        Args:
            n_features: 輸入特徵數
            hidden_size: LSTM 隱藏層維度
            num_layers: LSTM 層數
            dropout: Dropout 比率
            seq_len: 序列長度
            use_attention: 是否使用注意力機制
        """
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.use_attention = use_attention
        
        # LSTM 編碼器
        self.lstm = LSTMEncoder(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # 注意力層 (可選)
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # 輸出層
        self.output_layer = nn.Sequential(
            # 第一層：先放寬或保留充裕的特徵空間
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 第二層：平緩遞減到 hidden_size
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 輸出層
            nn.Linear(hidden_size, 1)
        )
        
        # Residual shortcut
        self.shortcut = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: (batch, seq_len, n_features)
            
        Returns:
            torch.Tensor: (batch,) 預測值
        """
        # LSTM 編碼
        lstm_output, (h_n, c_n) = self.lstm(x)  # lstm_output: (batch, seq_len, hidden_size)
        
        # 特徵提取
        if self.use_attention:
            # 使用注意力加權
            features = self.attention(lstm_output)  # (batch, hidden_size)
        else:
            # 使用最後一個時間步的輸出
            features = lstm_output[:, -1, :]  # (batch, hidden_size)
        
        # 輸出 (附加 Residual Connection)
        out = self.output_layer(features) + self.shortcut(features)  # (batch, 1)
        
        return out.squeeze(-1)


class RegimeEmbeddingLSTM(nn.Module):
    """
    含 Regime Embedding 的 LSTM 模型
    """
    
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 seq_len: int = 60, n_regimes: int = 2, use_attention: bool = True):
        """
        初始化模型
        
        Args:
            n_features: 輸入特徵數
            hidden_size: LSTM 隱藏層維度
            num_layers: LSTM 層數
            dropout: Dropout 比率
            seq_len: 序列長度
            n_regimes: regime 類別數 (預設 2: bull/bear)
            use_attention: 是否使用注意力機制
        """
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.n_regimes = n_regimes
        self.use_attention = use_attention
        
        # Regime Embedding (維度與特徵對齊以進行相加)
        self.regime_embedding = nn.Embedding(n_regimes, hidden_size)
        
        # LSTM 編碼器
        self.lstm = LSTMEncoder(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # 注意力層 (可選)
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # 輸出層 (使用相加融合 Conditioning)
        self.output_layer = nn.Sequential(
            # 第一層：維持 hidden_size 作為輸入並放寬
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 第二層：平緩遞減到 hidden_size
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 輸出層
            nn.Linear(hidden_size, 1)
        )
        
        # Residual shortcut
        self.shortcut = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: (batch, seq_len, n_features)
            regime: (batch,) regime 標籤
            
        Returns:
            torch.Tensor: (batch,) 預測值
        """
        # LSTM 編碼
        lstm_output, (h_n, c_n) = self.lstm(x)
        
        # 特徵提取
        if self.use_attention:
            features = self.attention(lstm_output)  # (batch, hidden_size)
        else:
            features = lstm_output[:, -1, :]  # (batch, hidden_size)
        
        # Regime Embedding
        regime_emb = self.regime_embedding(regime)  # (batch, hidden_size)
        
        # 相加特徵和 regime embedding (Conditioning 融合)
        combined = features + regime_emb  # (batch, hidden_size)
        
        # 輸出 (附加 Residual Connection)
        out = self.output_layer(combined) + self.shortcut(combined)  # (batch, 1)
        
        return out.squeeze(-1)




class RegimeSwitchLSTM(nn.Module):
    """
    做法 A：模型先用歷史序列自己判斷 bull/bear，再路由到 bull_model / bear_model
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 seq_len: int = 60, use_attention: bool = True):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.use_attention = use_attention

        self.router_lstm = LSTMEncoder(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )

        if use_attention:
            self.router_attention = AttentionLayer(hidden_size)

        self.router_head = nn.Sequential(
            # 第一層：先放寬或保留充裕的特徵空間
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 第二層：平緩遞減到 hidden_size
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            # 輸出層
            nn.Linear(hidden_size, 2)
        )

        # Router Residual shortcut
        self.router_shortcut = nn.Linear(hidden_size, 2)

        self.bull_model = TimeSeriesLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            use_attention=use_attention
        )

        self.bear_model = TimeSeriesLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            use_attention=use_attention
        )

    def _router_features(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.router_lstm(x)
        if self.use_attention:
            feat = self.router_attention(out)
        else:
            feat = out[:, -1, :]
        return feat

    def forward(self, x: torch.Tensor, regime_target: Optional[torch.Tensor] = None, return_aux: bool = False):
        router_feat = self._router_features(x)
        # 附加 Residual Connection
        regime_logits = self.router_head(router_feat) + self.router_shortcut(router_feat)

        # 計算牛熊市機率 
        regime_probs = torch.softmax(regime_logits, dim=1)

        bull_pred = self.bull_model(x)
        bear_pred = self.bear_model(x)

        # ====== 修改開始：導入 Hard Routing ======
        
        prob_bull = regime_probs[:, 0]
        prob_bear = regime_probs[:, 1]
        
        # 產生 0 或 1 的遮罩 (Mask)。如果 prob_bull >= prob_bear，is_bull 就是 1.0，否則是 0.0
        is_bull = (prob_bull >= prob_bear).float()
        is_bear = (prob_bear > prob_bull).float()
        
        # 使用遮罩進行硬切換：贏家全拿，輸家歸零
        reg_out = (is_bull * bull_pred) + (is_bear * bear_pred)
        
        # ====== 修改結束 ======

        direction_logits = reg_out

        if return_aux:
            return {
                'regression': reg_out,
                'direction_logits': direction_logits,
                'regime_logits': regime_logits,
                'predicted_regime': torch.argmax(regime_logits, dim=1)
            }

        return reg_out

    def predict_regime(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._router_features(x)
        # 附加 Residual Connection
        logits = self.router_head(feat) + self.router_shortcut(feat)
        return torch.argmax(logits, dim=1)

# === 向後相容性別名 (保持與舊代碼兼容) ===

# 基礎模型
TimeSeriesTransformer = TimeSeriesLSTM

# Regime Embedding 模型
RegimeEmbeddingTransformer = RegimeEmbeddingLSTM

# Regime Switch 模型
RegimeSwitchTransformer = RegimeSwitchLSTM


def create_model(model_type: str, n_features: int, config, 
                 device: str = 'cpu') -> nn.Module:
    """
    創建模型
    
    Args:
        model_type: 模型類型 ('single', 'regime_embedding', 'regime_switch')
        n_features: 輸入特徵數
        config: 配置
        device: 設備
        
    Returns:
        nn.Module: 模型
    """
    hidden_size = getattr(config.model, 'd_model', 64)
    num_layers = getattr(config.model, 'n_encoder_layers', 2)
    dropout = getattr(config.model, 'dropout', 0.2)
    seq_len = config.data.lookback_window
    
    if model_type == 'single':
        model = TimeSeriesLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            use_attention=True
        )
    elif model_type == 'regime_embedding':
        model = RegimeEmbeddingLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            use_attention=True
        )
    elif model_type == 'regime_switch':
        model = RegimeSwitchLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            use_attention=True
        )
    else:
        raise ValueError(f"未知的模型類型: {model_type}")
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """獲取模型摘要"""
    lines = []
    lines.append(f"模型類型: {model.__class__.__name__}")
    lines.append(f"參數數量: {count_parameters(model):,}")
    lines.append("")
    lines.append("模型結構:")
    lines.append(str(model))
    return "\n".join(lines)


if __name__ == "__main__":
    # 測試模型
    batch_size = 32
    seq_len = 60
    n_features = 10
    
    # 創建測試輸入
    x = torch.randn(batch_size, seq_len, n_features)
    regime = torch.randint(0, 2, (batch_size,))
    
    print("=" * 50)
    print("LSTM 模型測試")
    print("=" * 50)
    
    # 測試基礎模型
    print("\n1. TimeSeriesLSTM (基礎模型)")
    model1 = TimeSeriesLSTM(n_features=n_features, hidden_size=64, num_layers=2, seq_len=seq_len)
    out1 = model1(x)
    print(f"   輸入形狀: {x.shape}")
    print(f"   輸出形狀: {out1.shape}")
    print(f"   參數數量: {count_parameters(model1):,}")
    
    # 測試 Regime Embedding 模型
    print("\n2. RegimeEmbeddingLSTM (含 Regime Embedding)")
    model2 = RegimeEmbeddingLSTM(n_features=n_features, hidden_size=64, num_layers=2, seq_len=seq_len)
    out2 = model2(x, regime)
    print(f"   輸入形狀: x={x.shape}, regime={regime.shape}")
    print(f"   輸出形狀: {out2.shape}")
    print(f"   參數數量: {count_parameters(model2):,}")
    
    # 測試 Regime Switch 模型
    print("\n3. RegimeSwitchLSTM (Bull/Bear 分開)")
    model3 = RegimeSwitchLSTM(n_features=n_features, hidden_size=64, num_layers=2, seq_len=seq_len)
    out3 = model3(x, regime)
    print(f"   輸入形狀: x={x.shape}, regime={regime.shape}")
    print(f"   輸出形狀: {out3.shape}")
    print(f"   參數數量: {count_parameters(model3):,}")
    
    print("\n" + "=" * 50)
    print("✅ 所有模型測試通過!")
    print("=" * 50)
