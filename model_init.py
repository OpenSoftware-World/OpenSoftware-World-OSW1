import os
import re
import sys
import glob
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from config.model_config import *

NUM_THREADS = os.cpu_count() or 4
torch.set_num_threads(NUM_THREADS)
try:
    torch.set_num_interop_threads(max(1, NUM_THREADS // 2))
except RuntimeError:
    pass

# --- Automatic device selection: use a compatible GPU if available, otherwise fall back to CPU ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🧵 Number of CPU threads : {NUM_THREADS}")
print(f"🖥️  Selected device      : {DEVICE.type.upper()}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, model_path="opensoftware_world_osw1_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)

        if add_bos:
            ids = [self.bos_id] + ids

        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids):
        ids = [
            i for i in ids
            if i not in (self.pad_id, self.bos_id)
        ]

        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]

        return self.sp.decode(ids)

    def __len__(self):
        return self.sp.get_piece_size()

    @property
    def stoi(self):
        return {
            self.PAD: self.pad_id,
            self.UNK: self.unk_id,
            self.BOS: self.bos_id,
            self.EOS: self.eos_id,
        }

    @property
    def itos(self):
        return [
            self.sp.id_to_piece(i)
            for i in range(self.sp.get_piece_size())
        ]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be evenly divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(attn_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class OSW1Model(nn.Module):
    def __init__(self, vocab_size, cfg: dict, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.block_size = cfg["block_size"]

        self.tok_emb = nn.Embedding(vocab_size, cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["d_model"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_head"], cfg["d_ff"], cfg["dropout"])
            for _ in range(cfg["n_layer"])
        ])
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.85, top_k=40, eos_id=None):
        self.eval()
        # Make sure the input tensor lives on the same device as the model itself,
        # so generation works no matter which device the checkpoint was trained on.
        model_device = next(self.parameters()).device
        idx = idx.to(model_device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx

def find_checkpoint():
    candidates = glob.glob("opensoftware_world_osw1_*.pth")
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def load_checkpoint(path: str):
    print(f"📦 Loading: {path}")
    # map_location="cpu" guarantees the checkpoint can always be read back,
    # regardless of which device (GPU/MPS/CPU) it was trained on.
    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt["config"]
    vocab = Vocab("opensoftware_world_osw1_tokenizer.model")
    pad_id = vocab.sp.pad_id()

    model = OSW1Model(len(vocab), cfg, pad_id=pad_id).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    param_count = ckpt.get("param_count", sum(p.numel() for p in model.parameters()))
    training_time = ckpt.get("training_time_sec", None)
    final_loss = ckpt.get("final_loss", None)
    trained_on = ckpt.get("trained_on_device", "unknown")

    print("\n" + "=" * 64)
    print("🧠  OpenSoftware-World OSW1 — LOADED MODEL INFORMATION")
    print("=" * 64)
    print(f"  File                 : {path}")
    print(f"  Vocab size           : {len(vocab):,}")
    print(f"  Number of parameters : {param_count:,}")
    print(f"  d_model / n_layer    : {cfg['d_model']} / {cfg['n_layer']}")
    print(f"  n_head / d_ff        : {cfg['n_head']} / {cfg['d_ff']}")
    print(f"  Context window       : {cfg['block_size']}")
    if training_time is not None:
        print(f"  Training time        : {training_time/60:.2f} minutes")
    if final_loss is not None:
        print(f"  Final training loss  : {final_loss:.4f}")
    print(f"  Trained on device    : {trained_on}  ->  Running on: {DEVICE.type}")
    print("=" * 64 + "\n")

    return model, vocab, cfg

def chat_loop(model: OSW1Model, vocab: Vocab):
    print("=" * 64)
    print("💬 OpenSoftware-World-OSW1 ready! You can start chatting. Type 'exit' to quit.")
    print("=" * 64)

    model_device = next(model.parameters()).device
    eos_id = vocab.stoi[Vocab.EOS]
    bos_id = vocab.stoi[Vocab.BOS]

    while True:
        try:
            user_in = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_in.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        if not user_in:
            continue

        ids = [bos_id] + vocab.encode(user_in)
        x = torch.tensor([ids], dtype=torch.long, device=model_device)
        out = model.generate(x, max_new_tokens=init_max_new_tokens, temperature=init_temperature, top_k=init_top_k, eos_id=eos_id)
        answer_ids = out[0, len(ids):].tolist()
        answer = vocab.decode(answer_ids)
        print(f"OpenSoftware-World-OSW1: {answer if answer else '(...silence...)'}")

def main():
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        if not os.path.isfile(ckpt_path):
            print(f"❌ File not found: {ckpt_path}")
            sys.exit(1)
    else:
        ckpt_path = find_checkpoint()
        if ckpt_path is None:
            print(
                "❌ No checkpoint files found in the directory.\n"
                "   Please train a model using 'python3 model_training.py' or\n"
                "   specify a checkpoint file using 'python3 model_init.py <file_path>'."
            )
            sys.exit(1)

    model, vocab, cfg = load_checkpoint(ckpt_path)
    chat_loop(model, vocab)


if __name__ == "__main__":
    main()