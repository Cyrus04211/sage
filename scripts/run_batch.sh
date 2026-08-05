#!/bin/bash
# 批量并发运行三臂实验。用法:
#   bash scripts/run_batch.sh jobs.txt [并发数, 默认 12]
# jobs.txt 每行: <arm:A|B|C> <instance 相对路径> <seed>
# 每个任务单进程单线程; 峰值内存 ≈ 并发数 × ~3GB。
set -u
JOBS=${1:?usage: run_batch.sh jobs.txt [parallel]}
PAR=${2:-12}
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python; export PY   

# Arm B/C 使用的 LLM (可用环境变量覆盖, 默认 gpt-5.6-terra)
LLM_MODEL_ARG="--model ${SAGE_LLM_MODEL:-gpt-5.6-terra}"; export LLM_MODEL_ARG

# 可用环境变量: SAGE_LLM_MODEL, TRIALS (默认 20), RESUME=1 (续跑至 TRIALS 总预算)
TRIALS_ARG="--trials ${TRIALS:-20}"; export TRIALS_ARG
RESUME_ARG="${RESUME:+--resume}"; export RESUME_ARG

run_one() {
  arm=$1; inst=$2; seed=$3
  name=$(basename "$inst" .lp)
  case $arm in
    A) $PY -m sage.arms.run_arm_a --instance "$inst" $TRIALS_ARG \
          --time-limit 300 --seed "$seed" $RESUME_ARG \
          --out "results/arm_a/${name}_s${seed}.json" \
          > "results/log_arm_a_${name}_s${seed}.log" 2>&1 ;;
    B) $PY -m sage.arms.run_arm_b --instance "$inst" --trials 20 --n-ws 5 \
          --k 6 --time-limit 300 --seed "$seed" $LLM_MODEL_ARG \
          --out "results/arm_b/${name}_s${seed}.json" \
          > "results/log_arm_b_${name}_s${seed}.log" 2>&1 ;;
    C) $PY -m sage.arms.run_arm_b --instance "$inst" --trials 20 --n-ws 5 \
          --k 6 --time-limit 300 --seed "$seed" --rag $LLM_MODEL_ARG \
          --out "results/arm_c/${name}_s${seed}.json" \
          > "results/log_arm_c_${name}_s${seed}.log" 2>&1 ;;
    D) $PY -m sage.arms.run_arm_d --instance "$inst" $TRIALS_ARG \
          --time-limit 300 --seed "$seed" $RESUME_ARG \
          --out "results/arm_d/${name}_s${seed}.json" \
          > "results/log_arm_d_${name}_s${seed}.log" 2>&1 ;;
    E) $PY -m sage.arms.run_arm_e --instance "$inst" --trials 20 --n-ws 5 \
          --k 6 --time-limit 300 --seed "$seed" $LLM_MODEL_ARG \
          --out "results/arm_e/${name}_s${seed}.json" \
          > "results/log_arm_e_${name}_s${seed}.log" 2>&1 ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  echo "[done] $arm $name seed=$seed rc=$?"
}
export -f run_one

# 错峰启动: 每 20s 放一个任务, 避免同时读 500MB 模型文件
mkdir -p results
xargs -P "$PAR" -n 3 bash -c 'sleep $((RANDOM % 60)); run_one "$@"' _ < "$JOBS"
echo "ALL JOBS DONE"
