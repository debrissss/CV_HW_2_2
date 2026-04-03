#!/bin/bash

# ==============================================================================
# 批量执行实验三 (归一化)、实验四 (正则化) 和 实验七 (下采样)
# 用户可以直接运行此脚本来自动化完整的对比实验流程。
# ==============================================================================

# 设置项目根目录 (假设脚本在 scripts/ 目录下)
cd "$(dirname "$0")/.." || exit

# 定义需要运行的实验组
EXPERIMENTS=("exp3" "exp4" "exp7")

echo "--- 开始自动化实验流程 (Exp 3, 4, 7) ---"

for EXP in "${EXPERIMENTS[@]}"; do
    echo -e "\n>>> 正在执行实验组: ${EXP} <<<"
    
    # 获取该实验组的所有配置文件
    CONFIGS=$(ls configs/${EXP}_*.yaml 2>/dev/null)
    
    if [ -z "$CONFIGS" ]; then
        echo "警告: 未找到匹配 configs/${EXP}_*.yaml 的配置文件，跳过该组。"
        continue
    fi

    for CONFIG in $CONFIGS; do
        echo "正在运行配置: ${CONFIG}"
        # 执行训练 (不使用 --resume 以确保重新覆盖训练记录)
        python main.py --config "${CONFIG}"
    done

    # 训练完成后，自动执行汇总工具
    echo "正在汇总实验组 ${EXP} 的结果..."
    python tools/summarize.py --group "${EXP}"
done

echo -e "\n--- 所有实验执行完毕！汇总结果已保存至 results/ 目录下的各 summarize 文件夹。 ---"
