import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=['review', 'label'])
    
    # 保持标签比例采样
    _, sample_df = train_test_split(
        df, test_size=args.sample_size, random_state=args.seed, stratify=df['label']
    )
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    sample_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"成功采样 {len(sample_df)} 条数据至 {args.output_csv}")

if __name__ == "__main__":
    main()
