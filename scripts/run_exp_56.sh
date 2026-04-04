#!/bin/bash

# ==============================================================================
# 批量执行实验五 (残差连接) 和 实验六 (卷积核尺寸)
# ==============================================================================

# 进入项目根目录
cd "$(dirname "$0")/.." || exit

# 定义实验组
EXPERIMENTS=("exp5" "exp6")

echo "--- 开始自动化实验流程 (Exp 5, 6) ---"

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
        # 执行训练
        python main.py --config "${CONFIG}"
    done

    # 训练完成后，自动执行汇总工具
    echo "正在汇总实验组 ${EXP} 的结果..."
    python tools/summarize.py --group "${EXP}"
done

echo -e "\n--- 所有实验执行完毕！ ---"
