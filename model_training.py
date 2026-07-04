import os
import re
import json
import math
import time
import glob
import random
import contextlib

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from config.model_config import *

torch.manual_seed(42)
random.seed(42)

NUM_THREADS = os.cpu_count() or 4
torch.set_num_threads(NUM_THREADS)
try:
    torch.set_num_interop_threads(max(1, NUM_THREADS // 2))
except RuntimeError:
    # The number of interop threads can only be set once at the start of the program
    pass

try:
    torch.backends.mkldnn.enabled = True  # Intel MKL-DNN acceleration (if available)
except Exception:
    pass

# --- Automatic device selection: use a compatible GPU if available, otherwise fall back to CPU ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Autocasting to bfloat16 on the CPU can speed up most matmul operations (if supported)
# This optimization only matters when we are actually training on CPU.
USE_BF16_AUTOCAST = False
if DEVICE.type == "cpu":
    try:
        _ = torch.zeros(1, dtype=torch.bfloat16) + torch.zeros(1, dtype=torch.bfloat16)
        USE_BF16_AUTOCAST = True
    except Exception:
        USE_BF16_AUTOCAST = False

# Whether bfloat16 autocast is available on the current CUDA GPU (Ampere+ generally supports this)
USE_CUDA_BF16_AUTOCAST = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

print(f"🧵 Number of CPU threads           : {NUM_THREADS}")
print(f"🖥️  Selected training device    : {DEVICE.type.upper()}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"⚙️  bfloat16 autocast status    : "
      f"{'active (CPU)' if USE_BF16_AUTOCAST else ('active (CUDA)' if USE_CUDA_BF16_AUTOCAST else 'inactive')}")

@dataclass
class OSW1Config:
    data_dir: str = "data"

    block_size: int = training_block_size     
    d_model: int = training_d_model         
    n_layer: int = training_n_layer           
    n_head: int = training_n_head            
    d_ff: int = training_d_ff           
    dropout: float = training_dropout

    batch_size: int = training_batch_size
    grad_accum_steps: int = training_grad_accum_steps      
    epochs: int = training_epochs
    max_lr: float = training_max_lr
    min_lr: float = training_min_lr
    warmup_ratio: float = training_warmup_ratio
    weight_decay: float = training_weight_decay
    grad_clip: float = training_grad_clip
    label_smoothing: float = training_label_smoothing

    checkpoint_prefix: str = "opensoftware_world_osw1"

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

def load_json_pairs(json_dir):
    pairs = []
    if not os.path.isdir(json_dir):
        return pairs
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  {path} could not be read: {e}")
            continue
        intents = data.get("intents", data if isinstance(data, list) else [])
        for intent in intents:
            patterns = intent.get("patterns", []) or []
            responses = intent.get("responses", []) or []
            if not patterns or not responses:
                continue
            for p in patterns:
                for r in responses:
                    pairs.append((p, r))
    return pairs


def load_txt_qa_pairs(qa_dir):
    pairs = []
    if not os.path.isdir(qa_dir):
        return pairs
    for path in glob.glob(os.path.join(qa_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]
        q, a = None, None
        for raw in lines:
            line = raw.strip()
            if line.startswith("Q:"):
                q = line[2:].strip()
            elif line.startswith("A:"):
                a = line[2:].strip()
                if q is not None and a:
                    pairs.append((q, a))
                q, a = None, None
    return pairs


def load_plain_texts(txt_dir):
    texts = []
    if not os.path.isdir(txt_dir):
        return texts
    for path in glob.glob(os.path.join(txt_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
    return texts


def build_corpus(cfg: OSW1Config, vocab: Vocab):
    json_dir = os.path.join(cfg.data_dir, "json")
    qa_dir = os.path.join(cfg.data_dir, "txt_qa")
    txt_dir = os.path.join(cfg.data_dir, "txt")

    qa_pairs = load_json_pairs(json_dir) + load_txt_qa_pairs(qa_dir)
    plain_texts = load_plain_texts(txt_dir)

    print(f"📚 JSON + txt_qa pair count   : {len(qa_pairs)}")
    print(f"📄 Plain text file count     : {len(plain_texts)}")

    if not qa_pairs and not plain_texts:
        raise RuntimeError(
            "No data found! Please populate the 'data/json', 'data/txt', 'data/txt_qa' "
            "folders with data for the model to learn from."
        )

    sequences = []

    for q, a in qa_pairs:
        ids = [vocab.stoi[Vocab.BOS]]
        ids += vocab.encode(q)
        ids += vocab.encode(a)
        ids += [vocab.stoi[Vocab.EOS]]
        if len(ids) >= 4:
            sequences.append(ids)

    for t in plain_texts:
        ids = [vocab.stoi[Vocab.BOS]] + vocab.encode(t) + [vocab.stoi[Vocab.EOS]]
        stride = max(1, cfg.block_size // 2)
        for i in range(0, max(1, len(ids) - 1), stride):
            chunk = ids[i:i + cfg.block_size + 1]
            if len(chunk) >= 8:
                sequences.append(chunk)

    random.shuffle(sequences)
    print(f"🧩 Total training sequences (sequence): {len(sequences)}")
    print(f"🔤 Vocab size                    : {len(vocab)}")
    return sequences

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, block_size):
        self.sequences = sequences
        self.block_size = block_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx][: self.block_size + 1]
        return torch.tensor(ids, dtype=torch.long)


def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(x) for x in batch)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, : len(seq)] = seq
        x = padded[:, :-1].contiguous()
        y = padded[:, 1:].contiguous()
        return x, y
    return collate

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
    def __init__(self, vocab_size, cfg: OSW1Config, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_id,
                label_smoothing=self.cfg.label_smoothing,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=40, eos_id=None):
        was_training = self.training
        self.eval()
        # Make sure the input tensor lives on the same device as the model itself,
        # so generation works no matter which device the model was trained/loaded on.
        model_device = next(self.parameters()).device
        idx = idx.to(model_device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        if was_training:
            self.train()
        return idx

def count_parameters(model: OSW1Model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {
        "Token + Position Embedding": model.tok_emb.weight.numel() + model.pos_emb.weight.numel(),
        f"Transformer Blocks ({len(model.blocks)} pieces)": sum(p.numel() for p in model.blocks.parameters()),
        "Final LayerNorm": sum(p.numel() for p in model.ln_f.parameters()),
        "Output Layer (shared with embedding, no extra parameters)": 0,
    }
    return total, trainable, breakdown

def human_readable_param_count(n: int):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B", f"{max(1, round(n/1_000_000_000))}b"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M", f"{max(1, round(n/1_000_000))}m"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K", f"{max(1, round(n/1_000))}k"
    else:
        return str(n), str(n)

def print_model_report(model: OSW1Model, cfg: OSW1Config, vocab_size: int):
    total, trainable, breakdown = count_parameters(model)
    pretty, short = human_readable_param_count(total)
    size_mb = total * 4 / (1024 ** 2)

    print("\n" + "=" * 64)
    print("🧠  OpenSoftware-World OSW1 — MODEL REPORT")
    print("=" * 64)
    print(f"  Vocab size             : {vocab_size:,}")
    print(f"  Context window (block)  : {cfg.block_size}")
    print(f"  Embedding size (d_model)    : {cfg.d_model}")
    print(f"  Number of layers (n_layer)   : {cfg.n_layer}")
    print(f"  Head count (n_head)      : {cfg.n_head}")
    print(f"  Feed-forward size (d_ff)       : {cfg.d_ff}")
    print("-" * 64)
    for name, count in breakdown.items():
        print(f"  {name:<50}: {count:,}")
    print("-" * 64)
    print(f"  TOTAL PARAMETER COUNT   : {total:,}   (~{pretty})")
    print(f"  TRAINABLE PARAMETERS    : {trainable:,}")
    print(f"  Estimated model size      : {size_mb:.2f} MB (float32)")
    print(f"  Checkpoint file label       : {short}  ->  {cfg.checkpoint_prefix}_{short}.pth")
    print("=" * 64 + "\n")
    return short

def lr_at_step(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def get_autocast_context():
    """
    Returns the correct autocast context manager for whichever device we ended up
    training on (CUDA, CPU, or MPS/other). Falls back to a no-op context if the
    current device doesn't support (or benefit from) autocasting here.
    """
    if DEVICE.type == "cuda":
        dtype = torch.bfloat16 if USE_CUDA_BF16_AUTOCAST else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    elif DEVICE.type == "cpu" and USE_BF16_AUTOCAST:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()

def train(cfg: OSW1Config):
    vocab = Vocab()
    sequences = build_corpus(cfg, vocab)
    pad_id = vocab.stoi[Vocab.PAD]

    dataset = SeqDataset(sequences, cfg.block_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate(pad_id),
        num_workers=0,      
        drop_last=True,
    )

    if len(loader) == 0:
        raise RuntimeError(
            "The dataset is too small to even create a batch.  "
            "Try reducing 'batch_size' or adding more data."
        )

    model = OSW1Model(len(vocab), cfg, pad_id=pad_id).to(DEVICE)
    compiled_model = model
    try:
        compiled_model = torch.compile(model, backend="inductor")
        print("🚀 torch.compile has been enabled (provides an extra speed boost if available).")
    except Exception as e:
        print(f"ℹ️  torch.compile could not be used, continuing in normal mode: {e}")

    size_tag = print_model_report(model, cfg, len(vocab))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # Only needed for numerically-fragile float16 training on CUDA GPUs that lack
    # native bfloat16 support. When bfloat16 is available (or we're on CPU/MPS),
    # the scaler simply stays disabled and behaves as a no-op.
    use_grad_scaler = DEVICE.type == "cuda" and not USE_CUDA_BF16_AUTOCAST
    scaler = torch.amp.GradScaler(enabled=use_grad_scaler)

    steps_per_epoch = max(1, len(loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    print(f"⏱️  Total optimization steps : {total_steps} | Warmup steps: {warmup_steps}")
    print(f"🏋️  Training starting... ({cfg.epochs} epoch, batch={cfg.batch_size}, "
          f"grad_accum={cfg.grad_accum_steps})\n")

    global_step = 0
    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with get_autocast_context():
                _, loss = compiled_model(x, y)

            loss_scaled = loss / cfg.grad_accum_steps
            scaler.scale(loss_scaled).backward()

            if (i + 1) % cfg.grad_accum_steps == 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                lr = lr_at_step(global_step, total_steps, warmup_steps, cfg.max_lr, cfg.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - train_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"📈 Epoch {epoch:>3}/{cfg.epochs} | "
            f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s | total={elapsed_total/60:.1f}m"
        )

    total_time = time.time() - train_start
    print(f"\n✅ Training completed! Total time: "
          f"{total_time/60:.2f} minutes ({total_time:.1f} seconds)\n")

    ckpt_path = f"{cfg.checkpoint_prefix}_{size_tag}.pth"
    # Move every tensor in the state dict to CPU before saving. This makes the
    # checkpoint device-agnostic: a model trained on GPU can later be loaded
    # and run correctly on a machine that only has a CPU (and vice versa).
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "config": cfg.__dict__,
        "pad_id": pad_id,
        "param_count": sum(p.numel() for p in model.parameters()),
        "training_time_sec": total_time,
        "final_loss": avg_loss,
        "trained_on_device": DEVICE.type,
    }, ckpt_path)
    print(f"💾 Model saved: {ckpt_path}\n")

    return model, vocab, cfg, ckpt_path

def chat_loop(model: OSW1Model, vocab: Vocab, cfg: OSW1Config):
    print("=" * 64)
    print("💬 OSW1 with chat mode! Type 'exit' to quit.")
    print("=" * 64)
    model.eval()
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
        out = model.generate(x, max_new_tokens=training_max_new_tokens, temperature=training_temperature, top_k=training_top_k, eos_id=eos_id)
        answer_ids = out[0, len(ids):].tolist()
        answer = vocab.decode(answer_ids)
        print(f"OSW1: {answer if answer else '(...silence...)'}")

def load_checkpoint(path: str):
    # map_location="cpu" guarantees the checkpoint can always be read back,
    # regardless of which device it was trained on or whether a GPU is present
    # on the machine doing the loading.
    ckpt = torch.load(path, map_location="cpu")
    cfg = OSW1Config(**ckpt["config"])
    vocab = Vocab()
    model = OSW1Model(len(vocab), cfg, pad_id=ckpt["pad_id"])
    model.load_state_dict(ckpt["model_state_dict"])
    # Now move the freshly-loaded model onto whichever device is available
    # on *this* machine (GPU/MPS if present, otherwise CPU) so it runs correctly
    # no matter what device it was originally trained on.
    model.to(DEVICE)
    model.eval()
    trained_on = ckpt.get("trained_on_device", "unknown")
    print(f"📦 Checkpoint loaded (trained on: {trained_on}) -> running on: {DEVICE.type}")
    return model, vocab, cfg

def main():
    cfg = OSW1Config()
    model, vocab, cfg, ckpt_path = train(cfg)
    chat_loop(model, vocab, cfg)


if __name__ == "__main__":
    main()