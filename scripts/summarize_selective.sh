#!/bin/bash

# ==============================================================================
# 选择性汇总实验结果并开启特定放大图
# ==============================================================================

# 进入项目根目录
cd "$(dirname "$0")/.." || exit

echo "--- 开始选择性汇总实验结果 (Exp 1 - 7) ---"

# Exp 1: 默认汇总 (无放大图)
echo -e "\n>>> 汇总实验组: exp1 <<<"
python tools/summarize.py --group exp1

# Exp 2: 开启全部子图的放大图
echo -e "\n>>> 汇总实验组: exp2 (开启全部放大图) <<<"
python tools/summarize.py --group exp2 --show-train-inset --show-val-inset --show-acc-inset

# Exp 3: 仅开启训练 Loss 放大图 (子图一)
echo -e "\n>>> 汇总实验组: exp3 (开启训练 Loss 放大图) <<<"
python tools/summarize.py --group exp3 --show-train-inset

# Exp 4: 默认汇总 (无放大图)
echo -e "\n>>> 汇总实验组: exp4 <<<"
python tools/summarize.py --group exp4

# Exp 5: 默认汇总 (无放大图)
echo -e "\n>>> 汇总实验组: exp5 <<<"
python tools/summarize.py --group exp5

# Exp 6: 仅开启训练 Loss 放大图 (子图一)
echo -e "\n>>> 汇总实验组: exp6 (开启训练 Loss 放大图) <<<"
python tools/summarize.py --group exp6 --show-train-inset

# Exp 7: 仅开启训练 Loss 放大图 (子图一)
echo -e "\n>>> 汇总实验组: exp7 (开启训练 Loss 放大图) <<<"
python tools/summarize.py --group exp7 --show-train-inset

echo -e "\n--- 选择性汇总完毕！ ---"
