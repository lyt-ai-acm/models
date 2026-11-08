# Entropy-driven dynamic gating with N-best candidate fusion for robust sentiment analysis of noisy user-generated content

**Title:** Entropy-driven dynamic gating with N-best candidate fusion for robust sentiment analysis of noisy user-generated content

## Description

This repository provides the code and configuration for robust Chinese sentiment classification when user-generated text contains homophone noise. The workflow injects Chinese homophone noise, generates multiple correction candidates, reranks candidates with a KenLM language model, and combines N-best predictions with entropy-driven dynamic gating. It also includes baseline and supervised-contrastive training scripts, evaluation utilities, and experiment runners.

Large datasets, trained checkpoints, and KenLM binary models are not included. Paths to these artifacts must be supplied locally.

## Dataset Information

The primary data setting is Chinese user-generated sentiment text, such as Sina Weibo posts from the `weibo_senti_100k` dataset. The repository expects CSV files with the following default columns:

- `review`: Chinese text, review, comment, or post content.
- `label`: sentiment label. The default three-class convention is `0 = negative`, `1 = neutral`, and `2 = positive`.

A binary dataset may be used directly, or an auxiliary neutral dataset may be merged and assigned label `1` for three-class experiments. Do not commit private, restricted, or large raw datasets to this repository.

Expected layout:

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

The generated N-best CSV contains:
- `id`: sample identifier
- `orig`: original/noisy input text
- `cand_1 ... cand_k`: correction candidates
- `w_1 ... w_k`: candidate weights
- `score_1 ... score_k`: candidate scores


## Code Information

- `scripts/inject_noise.py`: inject character-level homophone noise into a CSV file.
- `pipeline/run_jieba.py` and `pipeline/run_pkuseg.py`: generate N-best correction candidates.
- `pipeline/normalize_top10.py`: candidate generation, scoring, beam search, and top-N normalization.
- `train/train_baseline.py`: baseline classifier training.
- `train/train_contrastive.py`: supervised-contrastive classifier training.
- `train/infer_with_nbest.py`: N-best inference, weighted aggregation, and entropy-based fallback.
- `scripts/run_matrix_experiment.sh`, `scripts/run_fewshot_experiment.sh`, and related scripts: reproducible experiment runners.
- `config/tunable_params_Version3.yaml`: tunable training, normalization, inference, and ablation parameters.
- `model.yml`: Conda environment specification.

## Usage Instructions

### 1. Clone the repository

```bash
git clone https://github.com/lyt-ai-acm/models.git
cd models
```

### 2. Create the environment

The supplied environment file targets Python 3.10. Create it with:

```bash
conda env create -f model.yml
conda activate <your-environment-name>
```

`model.yml` contains a machine-specific `prefix` and mirror channels. Adjust or remove those entries when using another operating system or package mirror.

### 3. Verify dependencies

Run:
python -c "import torch, transformers, datasets, accelerate, sentencepiece, evaluate, pandas, numpy, scipy, sklearn, jieba, pkuseg, pypinyin, kenlm, yaml; print('Dependencies OK')"

Expected output:
Dependencies OK

If this command fails, check that the conda environment from `model.yml` is activated.

### 4. Prepare data and models

Place train/dev/test CSV files under `data/splits/`, ensure that their text and label columns match the command-line options, copy the homophone dictionaries into `resources/`, and provide a compatible KenLM model such as `models/lm_jieba_3gram_20k.klm`.

### 5. Inject homophone noise

```bash
PYTHONPATH=. python scripts/inject_noise.py \
  --input_csv data/splits/train.csv \
  --output_csv data/splits/train_noisy.csv \
  --char_homo resources/chinese_homophone_char.txt \
  --text_col review \
  --noise_ratio 0.15 \
  --seed 42
```

### 6. Generate N-best candidates

```bash
PYTHONPATH=. python pipeline/run_pkuseg.py \
  --input_csv data/splits/test.csv \
  --output_csv outputs/test_top10.csv \
  --text_col review \
  --kenlm_path models/lm_jieba_3gram_20k.klm \
  --word_homo resources/chinese_homophone_word.txt \
  --char_homo resources/chinese_homophone_char.txt
```

The Jieba and PKUSeg entry points share the same normalization interface; select the script appropriate for the segmentation pipeline in your experiment.

### 7. Train a baseline model

```bash
PYTHONPATH=. python train/train_baseline.py \
  --backbone roberta_wwm_ext \
  --train_csv data/splits/train.csv \
  --dev_csv data/splits/dev.csv \
  --test_csv data/splits/test.csv \
  --output_dir outputs/baseline \
  --model_name hfl/chinese-roberta-wwm-ext \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --fp16
```

### 8. Train the supervised-contrastive model

```bash
PYTHONPATH=. python train/train_contrastive.py \
  --backbone roberta_wwm_ext \
  --train_csv data/splits/train.csv \
  --dev_csv data/splits/dev.csv \
  --test_csv data/splits/test.csv \
  --output_dir outputs/contrastive \
  --model_name hfl/chinese-roberta-wwm-ext \
  --epochs 3 \
  --batch_size 32 \
  --lr 2e-5 \
  --fp16 \
  --scl_weight 0.15
```

### 9. Run N-best inference and evaluation

```bash
PYTHONPATH=. python train/infer_with_nbest.py \
  --model_dir outputs/contrastive \
  --input_csv outputs/test_top10.csv \
  --truth_csv data/splits/test.csv \
  --out_json outputs/contrastive/metrics.json \
  --top_k 10 \
  --alpha 2.0 \
  --fallback_orig
```

Use `--entropy_low` and `--entropy_high` to change the boundaries for entropy-driven gating. The inference script writes metrics and prediction summaries to the JSON path specified by `--out_json`.

### 10. Run experiment suites

```bash
bash scripts/run_matrix_experiment.sh
bash scripts/run_fewshot_experiment.sh
```

Review each shell script before execution and update local data, model, and KenLM paths where necessary.

## Requirements

- Python 3.10.
- PyTorch, Transformers, Datasets, Accelerate, SentencePiece, and Evaluate.
- Pandas, NumPy, SciPy, scikit-learn, Matplotlib, and Seaborn.
- Jieba, PKUSeg, Pypinyin, and KenLM.
- PyYAML, regex, fsspec, tqdm, and related dependencies listed in `model.yml`.
- A shell environment for the experiment scripts.
- A CUDA-capable GPU is recommended for transformer training and inference, but CPU execution is possible for small tests.

`model.yml` is the authoritative dependency specification; no separate `requirements.txt` is currently provided.

## Methodology

1. Start with clean Chinese sentiment data and inject character-level homophone substitutions using a fixed random seed.
2. Segment noisy text and generate word-level and character-level correction candidates from the supplied homophone dictionaries.
3. Score candidates with KenLM probabilities, edit costs, and candidate priors, then retain an N-best list using beam search.
4. Train baseline or supervised-contrastive text classifiers on the selected data.
5. Run the classifier over the original text and N-best candidates.
6. Normalize candidate weights, aggregate the candidate predictions, and optionally interpolate with the original-text prediction.
7. Use normalized Shannon entropy to control the fallback strength: low entropy favors the correction candidates, while high entropy favors the original sentence.
8. Record accuracy, F1, precision, recall, confusion matrices, and experiment-specific comparison metrics.

Default seeds are set to `42` in the main scripts. Experiment runners cache generated N-best files and skip existing metric files where supported, which makes repeated runs faster and resumable.

## Citations

If you use the dataset, cite its original source and follow its licensing terms:

```bibtex
@misc{weibo_senti_100k,
  title        = {weibo_senti_100k: Chinese Sentiment Analysis Dataset},
  howpublished = {https://github.com/SophonPlus/ChineseNlpCorpus},
  note         = {Chinese sentiment analysis dataset derived from Sina Weibo}
}
```

If you use the pretrained encoder, also cite the corresponding model release, such as the `hfl/chinese-roberta-wwm-ext` model card. If this repository is used in a publication, cite the paper associated with this implementation:

> Entropy-driven dynamic gating with N-best candidate fusion for robust sentiment analysis of noisy user-generated content.

## License & Contribution Guidelines

No `LICENSE` file is currently included, so the repository does not yet declare an explicit redistribution license. Add a `LICENSE` file before distributing the code or derived artifacts under defined terms. Dataset and pretrained-model licenses remain the responsibility of their original providers.|
The repository currently does not include a LICENSE file and therefore does not yet declare explicit redistribution terms. Before distributing code or derived artifacts, add a LICENSE file with the intended terms. Dataset and pretrained model licenses remain with their original providers.

To contribute:

1. Fork the repository and create a focused feature branch.
2. Keep changes reproducible and document new data paths, parameters, and assumptions.
3. Run the relevant scripts or checks locally and avoid committing datasets, checkpoints, credentials, or other sensitive artifacts.
4. Open a pull request with a concise description, validation details, and any changes to expected input/output formats.
