#!/usr/bin/env bash
# Waits for arm 3 to finish, then runs the follow-on arms. Detached via nohup so
# it survives the controlling session.
REPO=/home/cc/spectrum-usage; R=$REPO/training/TSS-LCD/repro
U="--user $(id -u):$(id -g)"
Q=$R/results/queue.log
exec >> $Q 2>&1
echo "=== queue started $(date -u) ; waiting for theirs_random ==="
while docker ps --format '{{.Names}}' | grep -q theirs_random; do sleep 60; done
echo "=== arm 3 finished $(date -u); starting follow-ons ==="
cd $REPO
run () {  # run <tag> <extra args...>
  local tag=$1; shift
  [ -f $R/results/$tag/metrics.json ] && { echo "skip $tag (done)"; return; }
  echo "########## $tag  $(date -u) ##########"
  docker run --rm --gpus all $U --name $tag -v $REPO:/work -w /work \
    --entrypoint python tsslcd-ours:mpl -u training/TSS-LCD/repro/run_paper_faithful.py \
    --tag $tag --out /work/training/TSS-LCD/repro/results/$tag "$@" \
    2>&1 | tail -22
}
# 1. seed replicates - bound the run-to-run spread
run pf_chrono_s1 --split chrono --seed 1
run pf_chrono_s2 --split chrono --seed 2
run pf_random_s1 --split random --seed 1
run pf_random_s2 --split random --seed 2
# 2. smoothing ablation inside the paper-faithful model (leak isolated in ONE codebase)
run pf_chrono_smooth --split chrono --seed 0 --smooth
run pf_random_smooth --split random --seed 0 --smooth
echo "=== queue complete $(date -u) ==="
for f in $R/results/*/metrics.json; do
  python3 -c "import json,sys;d=json.load(open('$f'));print('%-20s mse=%9.4f pers=%8.4f lbm=%8.4f ae=%8.4f skill=%7.2f'%(d['tag'],d['model']['mse'],d['persistence']['mse'],d['lookback_mean']['mse'],d.get('autoencoder_ceiling',{}).get('mse',float('nan')),d['skill_vs_persistence']))" 2>/dev/null
done
