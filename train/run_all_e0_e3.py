# -*- coding: utf-8 -*-
import os
import subprocess


def run(cmd: str):
    print("\n[RUN]", cmd)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    ret = subprocess.call(cmd, shell=True, env=env)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    os.makedirs("outputs/norm", exist_ok=True)

    run("python scripts/01_split_weibo.py --input_csv data/Weibo_senti_100k.csv --out_dir data/splits")
    run("python scripts/02_tokenize_for_lm_jieba.py --input_csv data/splits/train.csv --output_txt data/lm/corpus_jieba.txt")

    print("bash scripts/03_train_kenlm.sh data/lm/corpus_jieba.txt models/lm_jieba_5gram")

    run(
        "python pipeline/run_jieba.py "
        "--input_csv data/splits/dev.csv "
        "--output_csv outputs/norm/dev_top10_jieba.csv "
        "--kenlm_path models/lm_jieba_5gram.klm "
        "--word_homo resources/chinese_homophone_word.txt "
        "--char_homo resources/chinese_homophone_char.txt"
    )

    run(
        "python train/train_roberta_binary.py "
        "--data_path data/Weibo_senti_100k.csv "
        "--output_dir outputs/roberta_binary_e0 "
        "--model_name hfl/chinese-roberta-wwm-ext "
        "--epochs 3 --batch_size 16 --lr 2e-5 --max_len 128 --seed 42"
    )

    run(
        "python train/infer_with_nbest.py "
        "--model_dir outputs/roberta_binary_e0/best_model "
        "--input_csv outputs/norm/dev_top10_jieba.csv "
        "--out_json outputs/roberta_binary_e0/e123_dev_metrics.json"
    )


if __name__ == "__main__":
    main()
