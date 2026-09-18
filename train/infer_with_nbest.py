# -*- coding: utf-8 -*-
import os
import json
import argparse
import inspect
import importlib
import importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--truth_csv", type=str, default="", help="真实带有label和文本的CSV文件")

    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--orig_col", type=str, default="orig")
    p.add_argument("--cand_prefix", type=str, default="cand_")
    p.add_argument("--w_prefix", type=str, default="w_")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--top_k", type=int, default=10, help="使用前K个候选")
    p.add_argument("--alpha", type=float, default=1.0, help="权重温度幂次: w^alpha 后再归一化")
    p.add_argument("--fallback_orig", action="store_true", help="启用门控回退到原句预测")

    # 保留旧超参接口防止外层Shell报错，但在内部被软门控取代或结合使用
    p.add_argument("--w1_threshold", type=float, default=0.35)
    p.add_argument("--margin_threshold", type=float, default=0.08)

    # 创新点专属超参: 熵的边界阈值
    p.add_argument("--entropy_low", type=float, default=0.20, help="熵低于此值完全相信纠错")
    p.add_argument("--entropy_high", type=float, default=0.70, help="熵高于此值完全回退原句")
    return p.parse_args()


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def normalize_weights(W, alpha=1.0):
    W = np.clip(W, 1e-12, None)
    if alpha != 1.0:
        W = np.power(W, alpha)
    Z = W.sum(axis=1, keepdims=True)
    Z = np.where(Z <= 0, 1.0, Z)
    return W / Z


def compute_shannon_entropy(W):
    W_safe = np.clip(W, 1e-12, 1.0)
    H = -np.sum(W_safe * np.log(W_safe), axis=1)
    return H


# ==========================================
# 混合模型加载逻辑
# ==========================================
def detect_model_backend(model_dir):
    if os.path.exists(os.path.join(model_dir, "model_meta.json")):
        return "classical"
    return "hf"


def _load_generic_training_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(repo_root, "train", "_generic_training.py")
    spec = importlib.util.spec_from_file_location("_generic_training", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_char_vocab(char_vocab_cls, vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 直接越过 load 实例化
    vocab = char_vocab_cls()
    vocab.stoi = data
    return vocab


def _load_classical_model(model_dir, device):
    module = _load_generic_training_module()
    with open(os.path.join(model_dir, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    backbone = meta.get("backbone", "")
    vocab = _load_char_vocab(module.CharVocab, os.path.join(model_dir, "vocab.json"))

    if backbone == "textcnn":
        model = module.TextCNNClassifier(
            vocab_size=vocab.size,
            embed_dim=meta.get("embed_dim", 256)
        )
    elif backbone == "bilstm_attn":
        model = module.BiLSTMAttnClassifier(
            vocab_size=vocab.size,
            embed_dim=meta.get("embed_dim", 256),
            hidden_dim=meta.get("hidden_dim", 256)
        )
    else:
        raise ValueError(f"Unsupported classical backbone '{backbone}'")

    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, vocab


# ==========================================
# 两种预测封装
# ==========================================
def predict_prob_hf(texts, tokenizer, model, device, batch_size=64, max_len=128):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def predict_prob_classical(texts, vocab, model, device, batch_size=64, max_len=128):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ids = [vocab.encode(t, max_len) for t in batch]
            x = torch.tensor(ids, dtype=torch.long, device=device)

            # 这里统一处理返回的 logits 格式
            outputs = model(x)
            # TextCNN/BiLSTM 原版代码里：return (logits, feat) if return_features else logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")

    if args.truth_csv:
        truth_df = pd.read_csv(args.truth_csv, encoding="utf-8-sig")
        if len(truth_df) == len(df):
            df[args.label_col] = truth_df[args.label_col].values
            if args.orig_col not in df.columns and "review" in truth_df.columns:
                df[args.orig_col] = truth_df["review"].values

    if args.label_col not in df.columns:
        raise KeyError(f"label col '{args.label_col}' not found in {args.input_csv} and no valid truth_csv provided.")

    cand_cols_all = [c for c in df.columns if c.startswith(args.cand_prefix)]
    w_cols_all = [c for c in df.columns if c.startswith(args.w_prefix)]
    if len(cand_cols_all) == 0 or len(w_cols_all) == 0:
        raise ValueError("No candidate or weight columns found.")

    max_cand = min(len(cand_cols_all), len(w_cols_all))
    K = min(args.top_k, max_cand)

    cand_cols = [f"{args.cand_prefix}{i}" for i in range(1, K + 1)]
    w_cols = [f"{args.w_prefix}{i}" for i in range(1, K + 1)]

    y_true = df[args.label_col].astype(int).to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    # 智能装载模型并提供统一切口
    # ===============================
    backend = detect_model_backend(args.model_dir)
    if backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)
        predictor = lambda texts: predict_prob_hf(texts, tokenizer, model, device, args.batch_size, args.max_len)
    else:
        model, vocab = _load_classical_model(args.model_dir, device)
        predictor = lambda texts: predict_prob_classical(texts, vocab, model, device, args.batch_size, args.max_len)

    # ==== 预测原句（用于Base/回退）====
    if args.orig_col in df.columns:
        orig_texts = df[args.orig_col].astype(str).tolist()
    elif "review" in df.columns:
        orig_texts = df["review"].astype(str).tolist()
    else:
        orig_texts = None

    p_orig = None
    if orig_texts is not None:
        p_orig = predictor(orig_texts)

    # ==== 预测cand_1..cand_K ====
    P = []
    for c in cand_cols:
        texts = df[c].fillna("").astype(str).tolist()
        p = predictor(texts)
        P.append(p)
    P = np.stack(P, axis=1)  # [N, K]

    # ==== 权重处理 ====
    W = df[w_cols].fillna(0.0).astype(float).to_numpy()  # [N, K]
    Wn = normalize_weights(W, alpha=args.alpha)

    p_e1 = P[:, 0]
    y_e1 = (p_e1 >= 0.5).astype(int)

    p_e2 = P.mean(axis=1)
    y_e2 = (p_e2 >= 0.5).astype(int)

    p_e3 = (P * Wn).sum(axis=1)
    y_e3 = (p_e3 >= 0.5).astype(int)

    result = {}
    if p_orig is not None:
        y_base = (p_orig >= 0.5).astype(int)
        result["Base_orig"] = compute_metrics(y_true, y_base)

    result["E1_top1"] = compute_metrics(y_true, y_e1)
    result["E2_topk_avg"] = compute_metrics(y_true, y_e2)
    result["E3_topk_weighted"] = compute_metrics(y_true, y_e3)

    # ==== 【核心创新】E4: 熵驱动的柔性自适应门控 ====
    if args.fallback_orig:
        if p_orig is None:
            raise ValueError("fallback_orig=True but no orig/review column found in input csv.")

        H = compute_shannon_entropy(Wn)
        H_norm = H / np.log(K + 1e-12)

        lambda_fb = np.clip((H_norm - args.entropy_low) / (args.entropy_high - args.entropy_low), 0.0, 1.0)

        w1 = Wn[:, 0]
        hard_fallback_mask = (w1 < args.w1_threshold)
        lambda_fb[hard_fallback_mask] = 1.0

        p_e4_dynamic = lambda_fb * p_orig + (1.0 - lambda_fb) * p_e3
        y_e4_dynamic = (p_e4_dynamic >= 0.5).astype(int)

        m_dynamic = compute_metrics(y_true, y_e4_dynamic)
        m_dynamic["mean_normalized_entropy"] = float(H_norm.mean())
        m_dynamic["mean_lambda_fallback"] = float(lambda_fb.mean())

        result["E3_topk_weighted_fallback"] = m_dynamic
        result["E4_Entropy_Dynamic_Gating"] = m_dynamic

    result["_config"] = {
        "top_k": K,
        "alpha": args.alpha,
        "fallback_orig": bool(args.fallback_orig),
        "entropy_low": args.entropy_low,
        "entropy_high": args.entropy_high
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[Done] {args.out_json}")


if __name__ == "__main__":
    main()