#!/usr/bin/env bash
set -e  # 出错即停止

# ===== 核心路径配置 =====
ORIG_TRAIN="data/splits/train.csv"
TEST_DATA="data/splits/test.csv"
DEV_DATA="data/splits/dev.csv"
KENLM_PATH="models/lm_jieba_3gram_20k.klm"
CHAR_HOMO="resources/chinese_homophone_char.txt"
WORD_HOMO="resources/chinese_homophone_word.txt"
MODEL_NAME="hfl/chinese-roberta-wwm-ext"

# 精简后的维度 (2模型 x 3规模 x 3噪声 = 18组)
MODELS=("textcnn" "roberta_wwm_ext")
SIZES=(1000 3000 10000)
NOISES=(0.0 0.15 0.30)

EXP_ROOT="outputs/matrix_experiment"
mkdir -p "$EXP_ROOT"

SUMMARY="$EXP_ROOT/final_summary_optimized.csv"
if [ ! -f "$SUMMARY" ]; then
    echo "Model,Size,Noise,Baseline_F1,Ours_E4_F1,Delta" > "$SUMMARY"
fi

# 1. 【优化】全局只生成一次测试集 N-Best 候选 (11,999条测试数据处理较慢，缓存复用)
GLOBAL_TEST_NBEST="$EXP_ROOT/test_top10_global.csv"
if [ ! -f "$GLOBAL_TEST_NBEST" ]; then
    echo ">>> [OPTIMIZE] 正在生成全局测试集 N-Best 候选..."
    PYTHONPATH=. python pipeline/run_jieba.py \
        --input_csv "$TEST_DATA" --output_csv "$GLOBAL_TEST_NBEST" \
        --kenlm_path "$KENLM_PATH" --word_homo "$WORD_HOMO" --char_homo "$CHAR_HOMO"
fi

for size in "${SIZES[@]}"; do
    for noise in "${NOISES[@]}"; do
        echo "=========================================================="
        echo ">>> 正在执行: Size=$size | Noise=$noise"
        echo "=========================================================="
        
        DATA_DIR="$EXP_ROOT/size_${size}_noise_${noise}/data"
        mkdir -p "$DATA_DIR"

        # A. 采样与注噪 (仅在不存在时执行)
        TRAIN_FINAL="$DATA_DIR/train_noisy.csv"
        if [ ! -f "$TRAIN_FINAL" ]; then
            TMP_CLEAN="$DATA_DIR/train_clean.csv"
            PYTHONPATH=. python scripts/sample_data.py --input_csv "$ORIG_TRAIN" --output_csv "$TMP_CLEAN" --sample_size "$size"
            PYTHONPATH=. python scripts/inject_noise.py --input_csv "$TMP_CLEAN" --output_csv "$TRAIN_FINAL" --char_homo "$CHAR_HOMO" --noise_ratio "$noise"
        fi

        for model_key in "${MODELS[@]}"; do
            RUN_DIR="$EXP_ROOT/size_${size}_noise_${noise}/$model_key"
            
            # 【断点续传】如果 E4 最终评测结果已存在，则跳过该模型的训练和评测
            if [ -f "$RUN_DIR/ours/metrics_e4.json" ]; then
                echo ">>> [SKIP] $model_key 已完成"
                continue
            fi
            
            mkdir -p "$RUN_DIR/baseline" "$RUN_DIR/ours"

            # B. 训练 Baseline (使用 32 Batch + FP16 提速)
            echo ">>> [TRAIN] $model_key Baseline..."
            PYTHONPATH=. python train/train_baseline.py \
                --backbone "$model_key" \
                --train_csv "$TRAIN_FINAL" --dev_csv "$DEV_DATA" --test_csv "$TEST_DATA" \
                --output_dir "$RUN_DIR/baseline" \
                --model_name "$MODEL_NAME" \
                --epochs 3 --batch_size 32 --lr 2e-5 --fp16

            # C. 训练 Ours (监督对比学习 SupCon + 交叉熵 CE)
            echo ">>> [TRAIN] $model_key Ours (SupCon)..."
            PYTHONPATH=. python train/train_contrastive.py \
                --backbone "$model_key" \
                --train_csv "$TRAIN_FINAL" --dev_csv "$DEV_DATA" --test_csv "$TEST_DATA" \
                --output_dir "$RUN_DIR/ours" \
                --model_name "$MODEL_NAME" \
                --epochs 3 --batch_size 32 --lr 2e-5 --fp16 --scl_weight 0.15

            # D. 推理与汇总
            # 【修复】使用 Python 稳健地读取 JSON 中的 f1 分数，防止 grep 失败导致脚本崩溃
            BASE_F1=$(python -c "import json; d=json.load(open('$RUN_DIR/baseline/metrics.json')); print(f\"{d.get('test', {}).get('f1') or d.get('f1') or 0.0:.6f}\")")
            
            PYTHONPATH=. python train/infer_with_nbest.py \
                --model_dir "$RUN_DIR/ours/best_model" \
                --input_csv "$GLOBAL_TEST_NBEST" \
                --truth_csv "$TEST_DATA" \
                --out_json "$RUN_DIR/ours/metrics_e4.json" \
                --alpha 2.0 --fallback_orig
            
            # 【修复】使用 Python 从嵌套结构的 E4 结果中准确提取 F1
            OURS_F1=$(python -c "import json; d=json.load(open('$RUN_DIR/ours/metrics_e4.json')); print(f\"{d.get('E4_Entropy_Dynamic_Gating', {}).get('f1', 0.0):.6f}\")")
            
            # 计算提升值
            DELTA=$(python -c "print(f\"{$OURS_F1 - float($BASE_F1):.6f}\")")
            
            # 记录到 CSV
            echo "$model_key,$size,$noise,$BASE_F1,$OURS_F1,$DELTA" >> "$SUMMARY"
            echo ">>> [RESULT] $model_key | Size:$size | Noise:$noise | Delta:$DELTA"
        done
    done
done

echo "所有 18 组实验已完成！结果汇总见: $SUMMARY"
EOF