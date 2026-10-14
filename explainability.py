"""
Three-level explainability, computed from the model. Nothing here is drawn
from hardcoded shapes or random numbers.

  Spatial    : real Grad-CAM on ResNet34 layer4
  Temporal   : absolute prediction change after replacing visual frame features
  Structural : real GATConv attention coefficients (self-loops removed)

If any of these cannot be computed, the function raises. It never substitutes
a placeholder, because a placeholder shown as a research result is worse than
no result at all.
"""

import numpy as np
import torch
import torch.nn.functional as F

from model import landmark_name


# ----------------------------------------------------------------------
# Spatial: real Grad-CAM
# ----------------------------------------------------------------------

def compute_gradcam(model, video_tensor, landmarks_tensor, target_class=None,
                    device="cpu"):
    """
    True Grad-CAM over ResNet34's final convolutional block (layer4).

    Returns
        cams          : (T, 7, 7) normalised per-frame class activation maps
        pred_class    : int
        probs         : (2,) softmax probabilities
    """
    model.eval()
    video_tensor = video_tensor.to(device)
    landmarks_tensor = landmarks_tensor.to(device)

    target_layer = model.spatial_features[7]  # ResNet34 layer4

    activations, gradients = {}, {}

    def fwd_hook(_m, _i, out):
        activations["value"] = out
        out.register_hook(lambda g: gradients.setdefault("value", g))

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(video_tensor, landmarks_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. The target layer index is wrong "
                "for this architecture."
            )

        acts = activations["value"].detach()      # (T, 512, 7, 7)
        grads = gradients["value"].detach()       # (T, 512, 7, 7)
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)        # GAP over spatial dims
    cam = F.relu((weights * acts).sum(dim=1))             # (T, 7, 7)

    cam = cam.cpu().numpy()
    out = np.zeros_like(cam)
    for t in range(cam.shape[0]):
        c = cam[t]
        if c.max() > c.min():
            out[t] = (c - c.min()) / (c.max() - c.min())
    return out, target_class, probs


def upsample_cam(cam_2d, size):
    """Bilinear upsample a single (7,7) CAM to (H, W)."""
    t = torch.from_numpy(cam_2d).float()[None, None]
    up = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


# ----------------------------------------------------------------------
# Temporal: visual feature-replacement sensitivity
# ----------------------------------------------------------------------

def compute_temporal_attention(model, video_tensor, landmarks_tensor,
                               target_class=None, device="cpu"):
    """Normalized absolute score change after replacing one CNN feature
    with the sequence mean. GAT features remain fixed. This is sensitivity,
    not learned attention, frame deletion, or signed support for a class.
    A zero-sensitivity sequence returns zeros.
    """
    model.eval()
    video_tensor = video_tensor.to(device)
    landmarks_tensor = landmarks_tensor.to(device)
    b, t, c, h, w = video_tensor.size()

    with torch.no_grad():
        spatial = model.spatial_features(
            video_tensor.view(b * t, c, h, w)
        ).view(b, t, model.spatial_dim)

        edge_index = model.edge_index.to(device)
        structural = torch.stack(
            [model.gat(landmarks_tensor[0, ti], edge_index)[0] for ti in range(t)]
        ).mean(dim=0).unsqueeze(0)

        def classify(seq):
            lstm_out, _ = model.lstm(seq)
            fused = torch.cat([lstm_out[:, -1, :], structural], dim=1)
            return F.softmax(model.fc(fused), dim=1)[0]

        base = classify(spatial)
        if target_class is None:
            target_class = int(torch.argmax(base).item())
        base_p = float(base[target_class])

        mean_feat = spatial.mean(dim=1, keepdim=True)

        drops = np.zeros(t, dtype=np.float64)
        for ti in range(t):
            occluded = spatial.clone()
            occluded[:, ti, :] = mean_feat[:, 0, :]
            drops[ti] = abs(base_p - float(classify(occluded)[target_class]))

    total = drops.sum()
    if total <= 0:
        # every frame is interchangeable for this video; report flat rather
        # than dividing by zero or inventing structure
        return np.zeros(t), target_class
    return drops / total, target_class


# ----------------------------------------------------------------------
# Structural: real GAT attention
# ----------------------------------------------------------------------

def compute_gat_attention(model, landmarks_tensor, frame_idx=0, device="cpu"):
    """
    Extract GATConv layer-2 attention coefficients for one frame.

    GATConv adds self-loops internally, so the returned edge_index contains
    68 self-loop entries. Those are removed here: an edge from a landmark to
    itself is not a "landmark relationship" and reporting it as one is wrong.

    Also returns each edge's deviation from equal-neighbor weighting.
    This descriptive statistic does not measure localization or faithfulness.
    """
    model.eval()
    landmarks_tensor = landmarks_tensor.to(device)
    edge_index = model.edge_index.to(device)

    lm = landmarks_tensor[0, frame_idx]
    with torch.no_grad():
        x1 = F.elu(model.gat.gat1(lm, edge_index))
        _, (ei_out, att) = model.gat.gat2(x1, edge_index,
                                          return_attention_weights=True)

    ei = ei_out.cpu().numpy()
    a = att.detach().cpu().numpy().reshape(-1)

    keep = ei[0] != ei[1]              # drop self-loops
    ei_real, a_real = ei[:, keep], a[keep]

    # in-degree per destination node, counting the self-loop the layer added
    deg = np.zeros(68, dtype=np.float64)
    for d in ei[1]:
        deg[d] += 1
    uniform = np.array([1.0 / deg[d] for d in ei_real[1]])
    deviation = a_real - uniform

    edges = []
    for k in range(ei_real.shape[1]):
        s, d = int(ei_real[0, k]), int(ei_real[1, k])
        edges.append({
            "src": s,
            "dst": d,
            "label": f"{landmark_name(s)} -> {landmark_name(d)}",
            "attention": float(a_real[k]),
            "uniform": float(uniform[k]),
            "deviation": float(deviation[k]),
        })

    spread = float(np.abs(deviation).max()) if len(deviation) else 0.0
    stats = {
        "n_edges": int(ei_real.shape[1]),
        "att_min": float(a_real.min()) if len(a_real) else 0.0,
        "att_max": float(a_real.max()) if len(a_real) else 0.0,
        "att_mean": float(a_real.mean()) if len(a_real) else 0.0,
        "att_std": float(a_real.std()) if len(a_real) else 0.0,
        "max_abs_deviation": spread,
    }
    return edges, stats
