seed=42
sample_dir="/project/outputs/samples/samples_unconditioned/"
num_workers=8

runai submit \
  --name  controlnet-ssim-sample \
  --image aicregistry:5000/wds20:control_brain \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/compute_msssim_sample.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      num_workers=${num_workers}
