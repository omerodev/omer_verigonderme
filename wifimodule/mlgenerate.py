import os
import json
import random
import numpy as np
import torch
from torch import nn

# ---------------- Ayarlar ----------------
CLASSES = ["butterfly", "house", "star"]
MAX_SEQ_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SketchGenCond(nn.Module):
    def __init__(self, in_dim=5, hid=512, layers=2,
                 n_cls=len(CLASSES), emb=64, drop=0.3):
        super().__init__()
        # 1) Input projection
        self.input_fc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        # 2) Class embedding
        self.embed = nn.Embedding(n_cls, emb)
        # 3) LSTM
        self.lstm = nn.LSTM(hid + emb, hid, layers,
                            batch_first=True, dropout=drop)
        # 4) Norm + dropout
        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(drop)
        # 5) MLP heads
        self.fc_xy = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid//2, 2)
        )
        self.fc_pen = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid//2, 3)
        )
        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, strokes, cls_ids, hidden=None):
        # strokes: [B, T, 5]
        B, T, _ = strokes.size()
        x = self.input_fc(strokes)                    # [B, T, hid]
        e = self.embed(cls_ids).unsqueeze(1).expand(-1, T, -1)  # [B, T, emb]
        out, hidden = self.lstm(torch.cat([x, e], dim=2), hidden)
        out = self.drop(self.ln(out))
        xy = self.fc_xy(out)
        pen = self.fc_pen(out)
        return xy, pen, hidden


def sample_sequence(model, cls_id, max_len=MAX_SEQ_LEN):
    """
    model: SketchGenCond (trained),
    cls_id: int (class index),
    returns: list of [dx, dy, p1, p2, p3] steps
    """
    model.eval()
    cls = torch.tensor([cls_id], device=DEVICE)
    hidden = None
    # start token
    inp = torch.tensor([[[0, 0, 0, 1, 0]]], dtype=torch.float32, device=DEVICE)
    seq = []
    with torch.no_grad():
        for _ in range(max_len - 1):
            xy, pen_logits, hidden = model(inp, cls, hidden)
            last_xy = xy[0, -1]
            dx, dy = last_xy[0].item(), last_xy[1].item()
            probs = torch.softmax(pen_logits[0, -1], dim=0).cpu().numpy()
            pen = np.random.choice(3, p=probs)
            step = [
                dx, dy,
                1 if pen == 0 else 0,
                1 if pen == 1 else 0,
                1 if pen == 2 else 0
            ]
            seq.append(step)
            inp = torch.tensor([[[*step]]], dtype=torch.float32, device=DEVICE)
    model.train()
    return seq
