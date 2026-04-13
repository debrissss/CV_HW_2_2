#!/bin/bash

# ==============================================================================
# 批量重复执行实验七 (下采样/池化策略实验)
# 该脚本会遍历所有 exp7 的配置文件，并为每个配置执行 3 次独立实验。
# ==============================================================================

# 确保在项目根目录下执行
cd "$(dirname "$0")/.." || exit

# 设定实验组名称
EXP_GROUP="exp7"
REPEAT_TIMES=3

echo "============================================================"
echo "  正在启动实验组 ${EXP_GROUP} 的重复运行流程 (次数: ${REPEAT_TIMES})"
echo "============================================================"

# 获取匹配 configs/exp7_*.yaml 的所有文件
CONFIG_FILES=$(ls configs/${EXP_GROUP}_*.yaml 2>/dev/null)

if [ -z "$CONFIG_FILES" ]; then
    echo "错误: 未能在 configs/ 目录下找到 ${EXP_GROUP} 的配置文件。"
    exit 1
fi

for CONFIG in $CONFIG_FILES; do
    echo -e "\n>>> 处理配置文件: ${CONFIG} <<<"
    
    # 调用 main.py 并使用 --repeat 参数执行 3 次
    # 此模式会自动计算均值与方差，并将结果分类存放在 results/ 对应的 repeat 文件夹中
    python main.py --config "${CONFIG}" --repeat ${REPEAT_TIMES}
done

echo -e "\n============================================================"
echo "  实验组 ${EXP_GROUP} 的所有重复运行任务已完成！"
echo "  结果及统计摘要可在 results 目录下的对应文件夹及其 summary.txt 中查看。"
echo "============================================================"
