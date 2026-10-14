"""
DST-GAT model definition and checkpoint loading.

Final Year Project, Universiti Putra Malaysia
Student: Yogeswary (217219)
Supervisor: Dr. Siti Khadijah Binti Ali

Architecture is unchanged from training. What changed is the LOADING:
the previous version renamed GAT parameters and used strict=False, which
silently discarded the trained GAT projection weights. This version loads
strictly and refuses to start if anything is missing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GATConv


# ----------------------------------------------------------------------
# Facial graph (68 landmarks, anatomical connectivity)
# ----------------------------------------------------------------------

def build_facial_edge_index():
    """68-landmark anatomical graph, undirected (both directions stored)."""
    edges = []
    for i in range(16):          # jawline 0-16
        edges.append([i, i + 1])
    for i in range(17, 21):      # right eyebrow 17-21
        edges.append([i, i + 1])
    for i in range(22, 26):      # left eyebrow 22-26
        edges.append([i, i + 1])
    for i in range(27, 35):      # nose 27-35
        edges.append([i, i + 1])
    for i in range(36, 41):      # right eye 36-41 (closed)
        edges.append([i, i + 1])
    edges.append([41, 36])
    for i in range(42, 47):      # left eye 42-47 (closed)
        edges.append([i, i + 1])
    edges.append([47, 42])
    for i in range(48, 59):      # outer lip 48-59 (closed)
        edges.append([i, i + 1])
    edges.append([59, 48])
    for i in range(60, 67):      # inner lip 60-67 (closed)
        edges.append([i, i + 1])
    edges.append([67, 60])

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([ei, ei.flip(0)], dim=1)


LANDMARK_REGIONS = [
    (0, 16, "Jaw"),
    (17, 21, "R.Eyebrow"),
    (22, 26, "L.Eyebrow"),
    (27, 35, "Nose"),
    (36, 41, "R.Eye"),
    (42, 47, "L.Eye"),
    (48, 59, "OuterLip"),
    (60, 67, "InnerLip"),
]


def landmark_name(idx):
    for lo, hi, name in LANDMARK_REGIONS:
        if lo <= idx <= hi:
            return f"{name}-{idx}"
    return f"LM-{idx}"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class FacialLandmarkGAT(nn.Module):
    """Two-layer Graph Attention Network over facial landmarks."""

    def __init__(self, input_dim=3, hidden_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                            concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x, (edge_index_out, attention_weights) = self.gat2(
            x, edge_index, return_attention_weights=True
        )
        graph_features = torch.mean(x, dim=0)
        return graph_features, (edge_index_out, attention_weights)


class ProposedModel(nn.Module):
    """DST-GAT: Dual-Stream Spatio-Temporal Graph Attention Network."""

    def __init__(self, lstm_hidden_size=256, lstm_num_layers=2,
                 gat_hidden_size=128, gat_num_heads=4, dropout=0.3):
        super().__init__()

        resnet = models.resnet34(weights=None)
        self.spatial_features = nn.Sequential(*list(resnet.children())[:-1])
        self.spatial_dim = 512

        self.lstm = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        self.lstm_output_size = lstm_hidden_size * 2

        self.gat = FacialLandmarkGAT(
            input_dim=3,
            hidden_dim=gat_hidden_size,
            num_heads=gat_num_heads,
            dropout=dropout,
        )
        self.gat_output_size = gat_hidden_size

        fusion_size = self.lstm_output_size + self.gat_output_size
        self.fc = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        self.register_buffer("_edge_index", build_facial_edge_index(),
                             persistent=False)

    @property
    def edge_index(self):
        return self._edge_index

    def forward(self, videos, landmarks, return_internals=False):
        """
        videos:    (B, T, 3, 224, 224)
        landmarks: (B, T, 68, 3)
        """
        b, t, c, h, w = videos.size()

        videos_flat = videos.view(b * t, c, h, w)
        spatial_feats_flat = self.spatial_features(videos_flat)
        spatial_feats = spatial_feats_flat.view(b, t, self.spatial_dim)

        lstm_out, _ = self.lstm(spatial_feats)
        temporal_features = lstm_out[:, -1, :]

        edge_index = self._edge_index.to(videos.device)

        gat_features_list = []
        for bi in range(b):
            frame_feats = []
            for ti in range(t):
                gf, _ = self.gat(landmarks[bi, ti], edge_index)
                frame_feats.append(gf)
            gat_features_list.append(torch.stack(frame_feats).mean(dim=0))
        structural_features = torch.stack(gat_features_list)

        fused = torch.cat([temporal_features, structural_features], dim=1)
        logits = self.fc(fused)

        if return_internals:
            return logits, {"lstm_out": lstm_out, "spatial_feats": spatial_feats}
        return logits


# ----------------------------------------------------------------------
# Checkpoint loading (strict)
# ----------------------------------------------------------------------

class CheckpointLoadError(RuntimeError):
    pass


def load_model(model_path, device="cpu", verbose=True):
    """
    Load the trained DST-GAT checkpoint.

    Loads with strict=True. If any parameter is missing or unexpected this
    raises instead of silently running with randomly initialised layers.
    """
    model = ProposedModel(
        lstm_hidden_size=256,
        lstm_num_layers=2,
        gat_hidden_size=128,
        gat_num_heads=4,
        dropout=0.3,
    )

    device = torch.device(device if (device != "cuda" or torch.cuda.is_available())
                          else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise CheckpointLoadError("Checkpoint format not recognised.")

    # Diagnose first, so the error message is actionable.
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointLoadError(
            "Checkpoint does not match the model architecture.\n"
            f"  Missing (model needs, checkpoint lacks): {result.missing_keys}\n"
            f"  Unexpected (checkpoint has, model lacks): {result.unexpected_keys}\n"
            "Do NOT proceed. Running with strict=False here would leave those "
            "layers randomly initialised and the predictions would be invalid.\n"
            "If the mismatch is 'lin' vs 'lin_src/lin_dst', your installed "
            "torch-geometric version differs from the one used for training. "
            "Install a matching version rather than renaming parameters."
        )

    model.to(device)
    model.eval()

    meta = {
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "val_f1": checkpoint.get("val_f1") if isinstance(checkpoint, dict) else None,
        "val_acc": checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None,
        "device": str(device),
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    if verbose:
        print("Model loaded with strict=True (all weights verified).")
        print(f"  device      : {meta['device']}")
        print(f"  epoch       : {meta['epoch']}")
        print(f"  val F1      : {meta['val_f1']}")
        print(f"  parameters  : {meta['num_params']:,}")

    return model, meta
