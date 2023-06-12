seed=42
samples_dir="/project/outputs/samples/flair_to_t1w/"
test_ids="/project/outputs/ids/test.tsv"
num_workers=8

runai submit \
  --name  controlnet-perf-metrics \
  --image aicregistry:5000/wds20:control_brain \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
  --volume /nfs/home/wds20/datasets/Biobank/derivatives/2d_controlnet/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_controlnet_performance.py \
      seed=${seed} \
      samples_dir=${samples_dir} \
      test_ids=${test_ids} \
      num_workers=${num_workers}
