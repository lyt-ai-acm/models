# -*- coding: utf-8 -*-
import argparse
import os
import random
import pandas as pd
from pypinyin import lazy_pinyin

def load_char_homo(path: str):
    """加载同音字表: pinyin -> [char1, char2, ...]"""
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0].strip() # 拼音作为 key
            vals = [x.strip() for x in parts[1:] if x.strip()]
            table[key] = vals
    return table

def char_py(ch: str) -> str:
    x = lazy_pinyin(ch)
    return x[0] if x else ""

def inject_noise(text: str, char_table: dict, ratio: float) -> str:
    chars = list(str(text))
    for i, ch in enumerate(chars):
        if random.random() < ratio: # 以 ratio 的概率触发替换
            py = char_py(ch)
            if not py:
                continue
            cands = char_table.get(py, [])
            # 排除原字
            cands = [c for c in cands if c != ch]
            if cands:
                chars[i] = random.choice(cands)
    return "".join(chars)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="原始干净数据集")
    ap.add_argument("--output_csv", type=str, required=True, help="输出的噪声数据集")
    ap.add_argument("--char_homo", type=str, required=True, help="同音字字典路径")
    ap.add_argument("--text_col", type=str, default="review", help="文本列名")
    ap.add_argument("--noise_ratio", type=float, default=0.15, help="替换比例，默认 15%")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    if os.path.dirname(args.output_csv):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    char_table = load_char_homo(args.char_homo)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")

    print(f"Injecting noise (ratio={args.noise_ratio}) into {len(df)} records...")
    df[args.text_col] = df[args.text_col].apply(lambda x: inject_noise(x, char_table, args.noise_ratio))

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] Saved noisy dataset to {args.output_csv}")

if __name__ == "__main__":
    main()