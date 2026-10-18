#!/usr/bin/env python3
"""TypeSig: single-file, no-CLI Haskell type-signature inference.

Place train.csv and test.csv beside this script or under /kaggle/input.
Run with Python or a notebook cell; no command-line options are required.
The generated file is /kaggle/working/submission.csv on Kaggle and
./working/submission.csv elsewhere.

The model and tokenizer are trained entirely from the competition CSVs.
No pretrained assets, external corpora, Haskell tools, or network are used.
Dependencies: numpy, pandas, scipy, scikit-learn, sentencepiece, torch.
"""
from __future__ import annotations

from pathlib import Path

import collections
import csv
import json
import math
import numpy as np
import os
import pandas as pd
import random
import re
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from types import SimpleNamespace
from typing import Any

# -----------------------------------------------------------------------------
# USER SETTINGS: edit here if desired; never need command-line arguments.
# -----------------------------------------------------------------------------
DATA_DIR = Path.cwd()        # e.g., Path("/kaggle/input/typesig"), or leave None
MODE = "hybrid"           # "hybrid" (Transformer + retrieval), "retrieval", "neural"
PREDICT_ONLY = False      # True: reuse model checkpoint in working/typesig_artifacts

CONFIG = dict(
    mode=MODE, predict_only=PREDICT_ONLY, seed=42,
    epochs=18, patience=5, refit_epochs=2, lr=0.00025,
    batch_size=24, decode_batch_size=16, validation_predictions=350,
    beam=1, device="auto", vocab_size=3000,
    max_source=340, max_target=112,
    dim=256, heads=4, layers=3, ff=768, dropout=0.12,
)

TRAIN_FILENAME = 'public/train.csv'
TEST_FILENAME = 'public/test.csv'

def _resolve_csv_pair(directory: Path) -> tuple[Path, Path] | None:
    """Return the train and test file paths if both exist in the directory, else None."""
    train_path = directory / TRAIN_FILENAME
    test_path = directory / TEST_FILENAME
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path
    return None

def find_input_files() -> tuple[Path, Path]:
    """Find an actual pair of competition CSVs without additional datasets."""
    if DATA_DIR is not None:
        root = Path(DATA_DIR).expanduser()
        pair = _resolve_csv_pair(root) or _resolve_csv_pair(root / "public")
        if pair is None:
            raise FileNotFoundError(
                f"Expected {TRAIN_FILENAME} and {TEST_FILENAME} in {root}"
            )
        return pair


def get_working_dir() -> Path:
    """Kaggle uses /kaggle/working; elsewhere create ./working."""
    kaggle = Path("./")
    out = kaggle if kaggle.is_dir() else Path.cwd() / "working"
    out.mkdir(parents=True, exist_ok=True)
    return out


# The public metric is based on these tokens, not character edit distance.
TYPE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)*|->|=>|::|[\[\](),]|[^\s]", re.UNICODE)
CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9']*|[-!#$%&*+./<=>?@\\^|~:]+|\d+|[^\s]", re.UNICODE)


def tokens_of_type(text: str) -> list[str]:
    return TYPE_TOKEN_RE.findall(text)


def row_score(pred: str, target: str) -> float:
    a, b = tokens_of_type(pred), tokens_of_type(target)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    return common / (len(a) + len(b))  # 0.5 * (2*common/(len(a)+len(b)))


def average_score(pred: list[str], gold: list[str]) -> float:
    return float(np.mean([row_score(p, g) for p, g in zip(pred, gold)]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype_mapping = {
        "example_id": "str",
        "repo_id": "str",
        "name": "str",
        "before": "str",
        "definition": "str",
        "type": "str",
    }
    train = pd.read_csv(train_path, dtype=dtype_mapping, keep_default_na=False)
    test = pd.read_csv(test_path, dtype="str", keep_default_na=False)
    needed = {'example_id', 'repo_id', 'name', 'before', 'definition'}
    if not needed.issubset(train) or not needed.issubset(test) or 'type' not in train:
        raise ValueError('Missing required CSV columns.')
    if train['example_id'].duplicated().any() or test['example_id'].duplicated().any():
        raise ValueError('Duplicate input IDs.')
    if (train['type'].str.len() == 0).any():
        raise ValueError('Empty training labels.')
    return train.reset_index(drop=True), test.reset_index(drop=True)


def src_text(row: Any) -> str:
    return (f"<name> {row.name}\n<before> {row.before}\n"
            f"<definition> {row.definition}")


def retrieval_text(row: Any) -> str:
    # Give the definition a higher weight, while including before-context for domain types.
    definition = str(row.definition)
    name = str(row.name)
    before = str(row.before)
    return f"{name} {name} {name} \n {definition} \n {definition} \n {before[-1100:]}"


def name_key(row: Any) -> str:
    # Series.name is the row index; the actual function name is a column.
    return str(row['name'] if isinstance(row, pd.Series) else row.name).lower().strip()


class RetrievalModel:
    """Train-only sparse lexical memory, no external code/index."""

    def __init__(self) -> None:
        self.cmat = None
        self.names = None
        self.wmat = None
        self.types = None
        self.char_vec = TfidfVectorizer(
            analyzer='char', ngram_range=(3, 5), min_df=1,
            sublinear_tf=True, max_features=145000, dtype=np.float32,
        )
        self.word_vec = TfidfVectorizer(
            tokenizer=lambda s: CODE_TOKEN_RE.findall(s), token_pattern=None,
            lowercase=False, ngram_range=(1, 2), min_df=1,
            sublinear_tf=True, max_features=85000, dtype=np.float32,
        )

    def fit(self, train: pd.DataFrame) -> 'RetrievalModel':
        texts = [retrieval_text(row) for row in train.itertuples(index=False)]
        self.cmat = self.char_vec.fit_transform(texts).tocsr()
        self.wmat = self.word_vec.fit_transform(texts).tocsr()
        self.types = train['type'].tolist()
        self.names = train['name'].str.lower().str.strip().tolist()
        return self

    def predict(self, queries: pd.DataFrame, chunk: int = 56) -> tuple[list[str], np.ndarray, np.ndarray]:
        texts = [retrieval_text(row) for row in queries.itertuples(index=False)]
        char_q = self.char_vec.transform(texts).tocsr()
        word_q = self.word_vec.transform(texts).tocsr()
        result, scores, same = [], [], []
        for start in range(0, len(texts), chunk):
            end = min(len(texts), start + chunk)
            sims = (0.48 * (char_q[start:end] @ self.cmat.T)
                    + 0.52 * (word_q[start:end] @ self.wmat.T))
            dense = sims.toarray()
            # A modest preference for exact function-name matches.
            for j in range(end - start):
                qname = name_key(queries.iloc[start + j])
                # Efficient only over retrieved top candidates; no. 17k-name loop per query.
                top = np.argpartition(dense[j], -min(40, dense.shape[1]))[-min(40, dense.shape[1]):]
                for k in top:
                    if qname and qname == self.names[k]:
                        dense[j, k] += 0.045
            best = dense.argmax(axis=1)
            for j, k in enumerate(best):
                result.append(self.types[int(k)])
                scores.append(float(dense[j, k]))
                same.append(name_key(queries.iloc[start + j]) == self.names[int(k)])
        return result, np.array(scores), np.array(same, dtype=bool)


def make_group_split(data: pd.DataFrame, seed: int, ratio: float = .14) -> tuple[np.ndarray, np.ndarray]:
    groups = data['repo_id']
    if groups.nunique() < 2:
        # Only for tiny artificial smoke-test data; actual competition has 392 repositories.
        rng = np.random.default_rng(seed)
        inds = rng.permutation(len(data))
        cut = max(1, min(len(data) - 1, int(len(data) * ratio)))
        return inds[cut:], inds[:cut]
    splitter = GroupShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    tr, va = next(splitter.split(data, groups=groups))
    assert not set(groups.iloc[tr]) & set(groups.iloc[va])
    return tr, va


def choose_retrieval_threshold(
    scores: np.ndarray, same_names: np.ndarray, retrieval: list[str],
    neural: list[str], gold: list[str]
) -> tuple[float, float, float]:
    thresholds = np.arange(.3, 1.001, .05)
    best = (-1., .8, .65)
    for general in thresholds:
        for exact_name in thresholds:
            use = (scores >= general) | (same_names & (scores >= exact_name))
            preds = [retrieval[i] if use[i] else neural[i] for i in range(len(gold))]
            metric = average_score(preds, gold)
            if metric > best[0]:
                best = (metric, float(general), float(exact_name))
    return best[1], best[2], best[0]


def merge_predictions(
    retrieval: list[str], sims: np.ndarray, same_names: np.ndarray,
    neural: list[str], threshold: float, name_threshold: float
) -> list[str]:
    use = (sims >= threshold) | (same_names & (sims >= name_threshold))
    return [retrieval[i] if use[i] or not neural[i] else neural[i]
            for i in range(len(retrieval))]


def make_tokenizer(train: pd.DataFrame, path_prefix: Path, vocab_size: int) -> Any:
    import sentencepiece as spm

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Train the tokenizer only on the training fold (not held-out labels).
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as f:
        corpus = f.name
        for row in train.itertuples(index=False):
            f.write(src_text(row).replace('\x00', ' ') + '\n')
            f.write(str(row.type).replace('\x00', ' ') + '\n')
    try:
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=str(path_prefix),
            model_type='bpe', vocab_size=vocab_size,
            character_coverage=1.0, hard_vocab_limit=False,
            byte_fallback=True,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols=['<name>', '<before>', '<definition>'],
            num_threads=max(1, min(8, os.cpu_count() or 2)),
            max_sentence_length=15000,
        )
    finally:
        os.unlink(corpus)
    sp = spm.SentencePieceProcessor(model_file=str(path_prefix) + '.model')
    return sp


class TokenizedRows:
    def __init__(self, data: pd.DataFrame, sp: Any, max_source: int, max_target: int,
                 labeled: bool = True) -> None:
        self.rows = []
        self.lengths = []
        for row in data.itertuples(index=False):
            n = sp.encode(str(row.name), out_type=int)[:32]
            b = sp.encode(str(row.before), out_type=int)[-110:]
            # Keep beginning of the definition, which contains arguments and key operations.
            remain = max(12, max_source - len(n) - len(b) - 4)
            d = sp.encode(str(row.definition), out_type=int)[:remain]
            source = ([sp.piece_to_id('<name>')] + n +
                      [sp.piece_to_id('<before>')] + b +
                      [sp.piece_to_id('<definition>')] + d + [sp.eos_id()])
            source = source[:max_source - 1] + [sp.eos_id()] if len(source) > max_source else source
            target = ([sp.bos_id()] + sp.encode(str(row.type), out_type=int)[:max_target - 2]
                      + [sp.eos_id()]) if labeled else None
            self.rows.append((source, target))
            self.lengths.append(len(source))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Any:
        return self.rows[idx]


def pack_batch(rows: list[tuple[list[int], list[int] | None]]) -> tuple[Any, Any]:
    import torch
    max_s = max(len(x[0]) for x in rows)
    sources = torch.zeros((len(rows), max_s), dtype=torch.long)
    targets = None
    if rows[0][1] is not None:
        max_t = max(len(x[1]) for x in rows)
        targets = torch.zeros((len(rows), max_t), dtype=torch.long)
    for i, (s, t) in enumerate(rows):
        sources[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        if t is not None:
            targets[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return sources, targets


def batches(dataset: TokenizedRows, batch_size: int, shuffle: bool, seed: int = 0):
    """Length buckets reduce padding and quadratic attention memory."""
    from torch.utils.data import DataLoader, Sampler

    class BucketSampler(Sampler):
        def __init__(self, data, bs, do_shuffle, s):
            self.data, self.bs, self.do_shuffle, self.seed = data, bs, do_shuffle, s

        def __iter__(self):
            rng = np.random.default_rng(self.seed)
            inds = np.arange(len(self.data))
            if self.do_shuffle:
                rng.shuffle(inds)
            chunks = []
            for offset in range(0, len(inds), self.bs * 40):
                block = sorted(inds[offset:offset + self.bs * 40], key=lambda i: self.data.lengths[i])
                chunks.extend([block[i:i+self.bs] for i in range(0, len(block), self.bs)])
            if self.do_shuffle:
                rng.shuffle(chunks)
            yield from chunks

        def __len__(self):
            return math.ceil(len(self.data) / self.bs)

    return DataLoader(dataset, batch_sampler=BucketSampler(dataset, batch_size, shuffle, seed),
                      collate_fn=pack_batch, num_workers=0)


def choose_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_model(vocab: int, max_source: int, max_target: int,
                dim: int, heads: int, layers: int, ff: int, dropout: float):
    import torch
    from torch import nn

    class SignatureTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embed = nn.Embedding(vocab, dim, padding_idx=0)
            self.src_pos = nn.Embedding(max_source + 8, dim)
            self.tgt_pos = nn.Embedding(max_target + 8, dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(dim)
            self.net = nn.Transformer(
                d_model=dim, nhead=heads,
                num_encoder_layers=layers, num_decoder_layers=layers,
                dim_feedforward=ff, dropout=dropout, batch_first=True,
                norm_first=False,
            )
            self.output = nn.Linear(dim, vocab, bias=False)
            self.output.weight = self.token_embed.weight

        def embed(self, x: torch.Tensor, positions: nn.Embedding):
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            return self.dropout(self.token_embed(x) * self.scale + positions(pos))

        def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mask = src.eq(0)
            src_emb = self.embed(src, self.src_pos)
            memory = self.net.encoder(src_emb, src_key_padding_mask=mask)
            return memory, mask

        def decode(
            self,
            tokens: torch.Tensor,
            memory: torch.Tensor,
            memory_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            length = tokens.size(1)
            causal = torch.ones((length, length), dtype=torch.bool, device=tokens.device).triu(1)
            tgt_emb = self.embed(tokens, self.tgt_pos)
            tgt_mask = tokens.eq(0)
            out = self.net.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            return self.output(out)

        def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
            mem, mask = self.encode(src)
            return self.decode(tgt_in, mem, mask)

    model = SignatureTransformer()
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model


def clean_signature(text: str) -> str:
    """Repair harmless subword spacing; do not impose invalid type guesses."""
    text = text.replace('\n', ' ').replace('\t', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*-\s*>\s*', ' -> ', text)
    text = re.sub(r'\s*=\s*>\s*', ' => ', text)
    text = re.sub(r'\s*:\s*:\s*', ' :: ', text)
    text = re.sub(r"([A-Za-z_][A-Za-z_0-9']*)\s+\.\s+([A-Za-z_][A-Za-z_0-9']*)", r'\1.\2', text)
    return text.strip()[:1000]


@__import__('torch').inference_mode()
def neural_predict(model: Any, data: TokenizedRows, sp: Any, device: str,
                   batch_size: int, max_len: int, beam: int = 1) -> list[str]:
    import torch
    from torch import cat
    model.eval()
    predictions = []
    # Preserve test-row order: length-bucketed batches reorder samples, which
    # would silently attach valid predictions to the WRONG example_id.
    from torch.utils.data import DataLoader
    infer_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                              collate_fn=pack_batch, num_workers=0)
    for sources, _ in infer_loader:
        sources = sources.to(device)
        memory, mask = model.encode(sources)
        B = sources.size(0)
        if beam == 1:
            words = torch.full((B, 1), sp.bos_id(), dtype=torch.long, device=device)
            done = torch.zeros((B,), dtype=torch.bool, device=device)
            for _step in range(max_len - 1):
                logits = model.decode(words, memory, mask)[:, -1, :]
                nxt = logits.argmax(dim=-1)
                nxt = torch.where(done, sp.eos_id(), nxt)
                words = cat((words, nxt.unsqueeze(1)), dim=1)
                done |= nxt.eq(sp.eos_id())
                if bool(done.all()):
                    break
            selected = words[:, 1:].cpu().tolist()
        else:
            # Fully batched beam search, including EOS handling.
            K, V = beam, model.output.out_features
            memory = memory.repeat_interleave(K, dim=0)
            mask = mask.repeat_interleave(K, dim=0)
            words = torch.full((B, K, 1), sp.bos_id(), device=device, dtype=torch.long)
            scores = torch.full((B, K), -1e9, device=device)
            scores[:, 0] = 0
            finished = torch.zeros((B, K), device=device, dtype=torch.bool)
            for _step in range(max_len - 1):
                logits = model.decode(words.reshape(B*K, -1), memory, mask)[:, -1, :]
                logprob = logits.log_softmax(-1).view(B, K, V)
                # Once finished, retain beam and append only EOS at zero cost.
                logprob = logprob.masked_fill(finished.unsqueeze(-1), -1e9)
                logprob[:, :, sp.eos_id()] = torch.where(
                    finished, torch.zeros_like(scores), logprob[:, :, sp.eos_id()])
                cand = (scores.unsqueeze(-1) + logprob).view(B, K*V)
                scores, indexes = cand.topk(K, dim=1)
                parent, nxt = indexes // V, indexes % V
                words = cat((torch.gather(words, 1, parent.unsqueeze(-1).expand(-1, -1, words.size(-1))),
                             nxt.unsqueeze(-1)), dim=-1)
                finished = torch.gather(finished, 1, parent) | nxt.eq(sp.eos_id())
                if bool(finished.all()):
                    break
            # Length normalization discourages implausibly short outputs.
            eos = words.eq(sp.eos_id())
            length = torch.where(eos.any(-1), eos.int().argmax(-1), words.size(-1) - 1).clamp(min=1)
            ranked = scores / length.float().pow(.65)
            best = ranked.argmax(dim=1)
            selected = words[torch.arange(B, device=device), best, 1:].cpu().tolist()
        for seq in selected:
            if sp.eos_id() in seq:
                seq = seq[:seq.index(sp.eos_id())]
            seq = [x for x in seq if x not in (sp.pad_id(), sp.bos_id())]
            predictions.append(clean_signature(sp.decode(seq)))
    return predictions


def teacher_loss(model: Any, loader: Any, device: str, train: bool,
                 optimizer=None, scaler=None, grad_clip: float = 1.0) -> float:
    import torch
    import torch.nn.functional as F
    model.train(train)
    total_loss = 0.
    total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            # CUDA autocast only; retain float32 for CPU/MPS reliability.
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=device == 'cuda'):
                output = model(src, tgt[:, :-1])
                gold = tgt[:, 1:]
                loss_sum = F.cross_entropy(output.reshape(-1, output.size(-1)),
                                           gold.reshape(-1), ignore_index=0,
                                           reduction='sum')
                count = int(gold.ne(0).sum())
                loss = loss_sum / max(1, count)
            if train:
                if scaler is not None and device == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        total_loss += float(loss_sum.detach().float().cpu())
        total_tokens += count
    return total_loss / max(1, total_tokens)


def train_neural(args: Any, train: pd.DataFrame, tr_idx: np.ndarray,
                 val_idx: np.ndarray, device: str):
    import torch
    import sentencepiece as spm
    from torch import optim

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_prefix = model_dir / 'typesig_sp'
    if args.predict_only:
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_prefix)+'.model')
        config = json.loads((model_dir/'config.json').read_text())
    else:
        sp = make_tokenizer(train.iloc[tr_idx], tokenizer_prefix, args.vocab_size)
        config = dict(vocab=sp.get_piece_size(), max_source=args.max_source,
                      max_target=args.max_target, dim=args.dim, heads=args.heads,
                      layers=args.layers, ff=args.ff, dropout=args.dropout)
        (model_dir/'config.json').write_text(json.dumps(config, indent=2))
    model = build_model(**config).to(device)
    bestfile = model_dir / 'best.pt'
    if args.predict_only:
        model.load_state_dict(torch.load(bestfile, map_location=device, weights_only=True))
        return model, sp, config

    train_rows = TokenizedRows(train.iloc[tr_idx], sp, args.max_source, args.max_target)
    valid_rows = TokenizedRows(train.iloc[val_idx], sp, args.max_source, args.max_target)
    optimiz = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.015,
                          betas=(.9, .98))
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    best_loss, patience = float('inf'), 0
    for epoch in range(args.epochs):
        # Cosine decay per epoch; simple and robust without external trainers.
        rate = args.lr * (.12 + .88 * (1 + math.cos(math.pi * epoch / max(1,args.epochs))) / 2)
        for g in optimiz.param_groups:
            g['lr'] = rate
        trainloss = teacher_loss(model, batches(train_rows, args.batch_size, True, args.seed+epoch),
                                 device, True, optimiz, scaler)
        validloss = teacher_loss(model, batches(valid_rows, args.batch_size, False), device, False)
        print(f'epoch {epoch+1:02d}/{args.epochs}: train NLL {trainloss:.4f}, '
              f'valid NLL {validloss:.4f} (lr {rate:.2g})', flush=True)
        if validloss < best_loss - 0.001:
            best_loss, patience = validloss, 0
            torch.save(model.state_dict(), bestfile)
        else:
            patience += 1
        if patience >= args.patience:
            print('Early stopping: validation NLL stopped improving.')
            break
    model.load_state_dict(torch.load(bestfile, map_location=device, weights_only=True))
    return model, sp, config


def validate_output(frame: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(frame.columns) != ['example_id', 'type']:
        raise ValueError('Submission columns/order must be example_id,type.')
    if len(frame) != len(test) or frame['example_id'].duplicated().any():
        raise ValueError('Submission count or IDs duplicated.')
    if set(frame['example_id']) != set(test['example_id']):
        raise ValueError('Submission ID set does not match test exactly.')
    if frame['type'].isna().any() or frame['type'].map(lambda x: not isinstance(x, str) or len(x.strip()) == 0 or len(x) > 1000).any():
        raise ValueError('Empty, missing, or overlong prediction.')


def main() -> None:
    args = SimpleNamespace(**CONFIG)
    if args.mode not in {"retrieval", "hybrid", "neural"}:
        raise ValueError("MODE must be retrieval, hybrid, or neural")
    if args.beam < 1:
        raise ValueError("CONFIG['beam'] must be >= 1")

    train_path, test_path = find_input_files()
    working_dir = get_working_dir()
    args.train = str(train_path)
    args.test = str(test_path)
    args.output = str(working_dir / "submission.csv")
    args.model_dir = str(working_dir / "typesig_artifacts")
    print(f"Training CSV: {train_path}")
    print(f"Test CSV:     {test_path}")
    print(f"Output CSV:   {args.output}")
    set_seed(args.seed)

    if args.beam < 1:
        args.error('--beam must be >=1')
    train, test = read_data(args.train, args.test)
    print(f'{len(train)} training rows, {len(test)} test rows; '
          f'{train.repo_id.nunique()} train repositories, {test.repo_id.nunique()} test repositories')
    if set(train.repo_id) & set(test.repo_id):
        print('WARNING: train/test contain overlapping repository IDs (unexpected for the challenge).')
    tr_idx, va_idx = make_group_split(train, args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    gates_path = model_dir / 'retrieval_gates.json'
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.predict_only:
        print(f'Validation: {len(tr_idx)} training, {len(va_idx)} held-out rows, '
              'split by repository.')
        retriever_valid = RetrievalModel().fit(train.iloc[tr_idx])
        retrieval_val, sim_val, same_val = retriever_valid.predict(train.iloc[va_idx])
        ret_metric = average_score(retrieval_val, train.iloc[va_idx]['type'].tolist())
        print(f'Retrieval held-out metric: {ret_metric:.5f}')
    else:
        retrieval_val = []

    if args.mode == 'retrieval':
        # The nearest-neighbor baseline is simple, legitimate, and makes a valid CSV.
        full_retrieval = RetrievalModel().fit(train)
        pred, _, _ = full_retrieval.predict(test)
    else:
        import torch
        device = choose_device(args.device)
        if device == 'cpu':
            print('CPU training may be slow. Set MODE="retrieval" at the top '
                  'of the script for a fast baseline.', flush=True)
        print(f'Neural training/inference device: {device}')
        model, sp, cfg = train_neural(args, train, tr_idx, va_idx, device)
        if args.predict_only:
            if gates_path.exists():
                gates = json.loads(gates_path.read_text())
                threshold = float(gates['threshold'])
                name_threshold = float(gates['name_threshold'])
            else:
                threshold, name_threshold = .85, .7
        else:
            val_sample = np.random.default_rng(args.seed).choice(
                len(va_idx), size=min(args.validation_predictions, len(va_idx)), replace=False)
            sample_train_positions = va_idx[val_sample]
            val_ds = TokenizedRows(train.iloc[sample_train_positions], sp,
                                   cfg['max_source'], cfg['max_target'], labeled=False)
            neural_val = neural_predict(model, val_ds, sp, device, args.decode_batch_size,
                                        cfg['max_target'], args.beam)
            gold_val = train.iloc[sample_train_positions]['type'].tolist()
            neural_metric = average_score(neural_val, gold_val)
            print(f'Neural sampled held-out metric ({len(gold_val)} rows): {neural_metric:.5f}')
            threshold, name_threshold, hybrid_score = choose_retrieval_threshold(
                sim_val[val_sample], same_val[val_sample],
                [retrieval_val[i] for i in val_sample], neural_val, gold_val)
            print(f'Hybrid sampled validation metric: {hybrid_score:.5f} '
                  f'(general similarity >= {threshold:.2f}, exact-name similarity >= {name_threshold:.2f})')
            gates_path.write_text(json.dumps(
                dict(threshold=threshold, name_threshold=name_threshold,
                     retrieval_metric=ret_metric, neural_sample_metric=neural_metric,
                     hybrid_sample_metric=hybrid_score, validation_size=len(gold_val)), indent=2))
            if args.refit_epochs > 0:
                print(f'Fine-tuning on ALL labeled rows for {args.refit_epochs} epoch(s).')
                all_ds = TokenizedRows(train, sp, cfg['max_source'], cfg['max_target'])
                optimiz = torch.optim.AdamW(model.parameters(), lr=args.lr * .2,
                                            weight_decay=.015)
                scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
                for e in range(args.refit_epochs):
                    loss = teacher_loss(model, batches(all_ds, args.batch_size, True, args.seed+500+e),
                                        device, True, optimiz, scaler)
                    print(f'full-data epoch {e+1}: train NLL {loss:.4f}', flush=True)
                torch.save(model.state_dict(), model_dir / 'best.pt')
        print('Generating neural test predictions...', flush=True)
        test_ds = TokenizedRows(test, sp, cfg['max_source'], cfg['max_target'], labeled=False)
        neural = neural_predict(model, test_ds, sp, device, args.decode_batch_size,
                                cfg['max_target'], args.beam)
        if args.mode == 'neural':
            pred = neural
        else:
            full_retrieval = RetrievalModel().fit(train)
            retrieval, sims, same = full_retrieval.predict(test)
            pred = merge_predictions(retrieval, sims, same, neural, threshold, name_threshold)
            nret = sum(x == y for x, y in zip(pred, retrieval))
            print(f'Hybrid selected a retrieval-matching prediction for {nret}/{len(pred)} rows.')

    # Every row has a nonempty, <=1000-character prediction, even if neural decoding fails.
    pred = [p if isinstance(p, str) and p.strip() else 'IO ()' for p in pred]
    pred = [p[:1000] for p in pred]
    submission = pd.DataFrame({'example_id': test['example_id'], 'type': pred})
    validate_output(submission, test)
    submission.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    # Verify file on disk, including preserved IDs and non-empty labels.
    saved = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    validate_output(saved, test)
    print(f'Saved {output_path.resolve()} ({len(saved)} rows); schema/IDs/length verified.')
    print('Preview:')
    print(saved.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
