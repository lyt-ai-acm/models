# Homophone-Robust Chinese Text Classification Models

This repository contains a Chinese text-classification workflow that studies robustness to homophone noise.  
It includes:
- homophone-based character noise injection,
- N-best candidate generation + KenLM reranking,
- baseline and supervised-contrastive training,
- inference/evaluation scripts and experiment runners.

## Repository Scope and Data Availability

- The repository contains code, configs, and small dictionary resources.
- **Large datasets and trained model artifacts are not included.**
- Several paths in scripts are project-specific (for example `data/splits/*.csv`, `models/lm_jieba_3gram_20k.klm`).

## Expected Data Layout

Many scripts assume this layout:

```text
data/
  splits/
    train.csv
    dev.csv
    test.csv
models/
  lm_jieba_3gram_20k.klm
resources/
  chinese_homophone_char.txt
  chinese_homophone_word.txt
```

### Expected CSV columns

- Text column: `review` (default in most scripts)
- Label column: `label` (default in training/evaluation scripts)

For generated N-best files (from `pipeline/run_jieba.py` or `pipeline/run_pkuseg.py`), columns typically include:
- `id`, `orig`
- `cand_1 ... cand_k`
- `w_1 ... w_k`
- `score_1 ... score_k`

## Code Organization

- `scripts/inject_noise.py`: injects homophone noise into text data.
- `pipeline/normalize_top10.py`: homophone normalization with candidate generation, KenLM scoring, beam search, and top-N output.
- `pipeline/run_jieba.py`, `pipeline/run_pkuseg.py`: build N-best candidate CSV files.
- `train/train_baseline.py`: baseline training entry point.
- `train/train_contrastive.py`: supervised-contrastive training entry point.
- `train/infer_with_nbest.py`: N-best inference and evaluation (including entropy/dynamic gating options).
- `scripts/run_matrix_experiment.sh`, `scripts/run_fewshot_experiment.sh`: matrix/few-shot style experiment runners with caching/resume logic.
- `config/tunable_params_Version3.yaml`: suggested tunable parameters and ablation combinations.

## Environment Requirements

Based on `model.yml` and `in.py` imports, the project expects:

- Python 3.10
- PyTorch + Transformers stack (`torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`)
- Data/science libs (`pandas`, `numpy`, `scikit-learn`, `scipy`)
- NLP libs (`pypinyin`, `jieba`, `pkuseg`, `kenlm`)
- Utility libs (`PyYAML`, `tqdm`, `regex`, `fsspec`)
- Shell environment for experiment scripts
- GPU is recommended for transformer training/inference

> Note: there is no `requirements.txt` in this repository. Use `model.yml` as the primary environment specification.

## Setup and Execution

### 1) Clone

```bash
git clone https://github.com/lyt-ai-acm/models.git
cd models
```

### 2) Create environment from `model.yml`

```bash
conda env create -f model.yml
conda activate <your-env-name>
```

`model.yml` includes a Windows-style `prefix` and mirror channels; adjust/remove them for your local setup as needed.

### 3) Verify dependencies

```bash
python in.py
```

### 4) Prepare data/resources/models

- Put split CSVs under `data/splits/`.
- Ensure `review` and `label` columns exist (or pass custom `--text_col` / `--label_col` where supported).
- Place homophone dictionaries in `resources/`.
- Place the KenLM model at the path used by your commands (for example `models/lm_jieba_3gram_20k.klm`).

## Representative Commands

### Inject homophone noise

```bash
PYTHONPATH=. python scripts/inject_noise.py \
  --input_csv data/splits/train.csv \
  --output_csv data/splits/train_noisy.csv \
  --char_homo resources/chinese_homophone_char.txt \
  --text_col review \
  --noise_ratio 0.15 \
  --seed 42
```

`noise_ratio` is the per-character probability of replacing a character with a randomly selected homophone candidate.

### Generate N-best candidates (jieba)

```bash
PYTHONPATH=. python pipeline/run_jieba.py \
  --input_csv data/splits/test.csv \
  --output_csv outputs/test_top10.csv \
  --kenlm_path models/lm_jieba_3gram_20k.klm \
  --word_homo resources/chinese_homophone_word.txt \
  --char_homo resources/chinese_homophone_char.txt
```

### Train baseline model

```bash
PYTHONPATH=. python train/train_baseline.py \
  --backbone roberta_wwm_ext \
  --train_csv data/splits/train.csv \
  --dev_csv data/splits/dev.csv \
  --test_csv data/splits/test.csv \
  --output_dir outputs/baseline \
  --model_name hfl/chinese-roberta-wwm-ext \
  --epochs 3 --batch_size 32 --lr 2e-5 --fp16
```

### Train supervised-contrastive model

```bash
PYTHONPATH=. python train/train_contrastive.py \
  --backbone roberta_wwm_ext \
  --train_csv data/splits/train.csv \
  --dev_csv data/splits/dev.csv \
  --test_csv data/splits/test.csv \
  --output_dir outputs/ours \
  --model_name hfl/chinese-roberta-wwm-ext \
  --epochs 3 --batch_size 32 --lr 2e-5 --fp16 --scl_weight 0.15
```

### N-best inference/evaluation

```bash
PYTHONPATH=. python train/infer_with_nbest.py \
  --model_dir outputs/ours/best_model \
  --input_csv outputs/test_top10.csv \
  --truth_csv data/splits/test.csv \
  --out_json outputs/ours/metrics_e4.json \
  --alpha 2.0 --fallback_orig
```

### Run experiment scripts

```bash
bash scripts/run_matrix_experiment.sh
bash scripts/run_fewshot_experiment.sh
```

## Implemented Methodology

The implemented workflow in this repository is:

1. Inject homophone-based character noise into training text.
2. Generate word-level and character-level candidate corrections.
3. Score candidates with character-level KenLM probability.
4. Use beam search and scoring terms (LM + prior - edit cost) to keep top-N candidates.
5. Train classification models (baseline and supervised contrastive variants).
6. Run N-best inference with weighted aggregation and optional entropy-based dynamic fallback to the original sentence.
7. Summarize F1 and Delta improvements in experiment outputs.

## Configuration and Tuning

See `config/tunable_params_Version3.yaml` for:
- recommended tunable training/data parameters,
- N-best/normalization parameters (`m`, `beam_size`, `topn`, `alpha`, `beta`, `lamb`, `delta`, `tau`),
- inference ablation settings (`top_k`, `alpha`, `fallback_orig`, thresholds),
- example ablation combinations.

## Reproducibility Notes

- Seed `42` is used by default in major scripts.
- Experiment shells cache global test N-best generation to avoid recomputation.
- Outputs are written under `outputs/` (for example `outputs/matrix_experiment/...`).
- Resume/skip behavior is implemented by checking existing metric files (for example `metrics_e4.json`).
- Results depend on local data/model paths and your environment setup.

## Dataset Information

The original test dataset used in this repository is a Chinese text dataset, not an English text dataset. In the default setting, the text data are Chinese user-generated reviews or posts, for example Chinese Weibo sentiment data. All preprocessing, homophone noise injection, candidate generation, normalization, and evaluation are designed for Chinese text.

The default label space is a three-class sentiment classification:

- 0 = negative
- 1 = neutral
- 2 = positive

The text column `review` means "user review", "comment", or "post content". It does not mean "censorship" or "examination" (shencha in pinyin). The label column `label` means the sentiment class. If the original data are binary (0/1), a separate neutral dataset can be merged and assigned label 1 (neutral) to form the three-class setting.

For N-best files, `orig` is the original Chinese sentence, which may contain homophone noise (tongyinzi in pinyin). `cand_1 ... cand_k` are Chinese correction candidates. `w_1 ... w_k` are their weights or confidences. `score_1 ... score_k` are normalization or language-model scores. These fields are not English text fields. They should be interpreted as Chinese text and Chinese-oriented correction scores.

`neg`, `neu`, and `pos` are abbreviations for negative, neutral, and positive. `Hanzi` in pinyin means Chinese characters. `Zhongwen` in pinyin means Chinese language or Chinese text.

## Citation

#### Dataset Citation

##### 1. weibo_senti_100k (Primary Dataset)

The primary dataset used in this repository is **weibo_senti_100k**, a publicly available Chinese sentiment analysis dataset derived from Sina Weibo. It contains over 100,000 manually annotated Weibo posts, with approximately 59,993 positive and 59,995 negative samples. The dataset covers diverse social media content, including daily user expressions, trending topic discussions, and short texts containing emoticons and internet slang.

In this repository, the dataset is used as a Chinese text classification corpus. The text column `review` means “user review”, “comment”, or “post content”; it does not mean “censorship” or “examination” (`shencha` in pinyin). The label column `label` contains the sentiment class. For the three-class setting, a separate neutral dataset can be merged and assigned label `1` (neutral).

**Source / Download Link:**

- GitHub: https://github.com/SophonPlus/ChineseNlpCorpus
- Dataset path in repository: `datasets/weibo_senti_100k`

**Suggested Citation Format:**

```bibtex
@misc{weibo_senti_100k,
  title        = {weibo\_senti\_100k: Chinese Sentiment Analysis Dataset},
  howpublished = {\url{https://github.com/SophonPlus/ChineseNlpCorpus}},
  note         = {Chinese sentiment analysis dataset from Sina Weibo}
}
```

##### 2. Neutral Dataset (Auxiliary Dataset)

If a separate neutral dataset is used to extend the binary setting to three classes, its source, license, and citation should follow the original dataset provider. The neutral data are assigned label `1` (neutral) in this repository.

## License

No explicit repository license file was detected at the time of writing.  
Please add a LICENSE file if you want to define redistribution and usage terms.

## Contributing

Practical contribution workflow:
1. Fork the repository and create a feature branch.
2. Keep changes minimal and focused.
3. Run relevant checks/scripts locally.
4. Open a pull request with clear description, data/path assumptions, and reproducibility notes.

## Limitations and Practical Notes

- Many scripts assume local/project-specific absolute or relative paths.
- Large datasets, checkpoints, and language models are intentionally not committed here.
- Do not commit sensitive data, credentials, or large binary artifacts to the repository.
