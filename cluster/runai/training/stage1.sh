seed=42
run_dir="aekl_v0"
training_ids="/project/outputs/ids/test.tsv"
validation_ids="/project/outputs/ids/test.tsv"
config_file="/project/configs/stage1/aekl_v0.yaml"
batch_size=128
n_epochs=50
adv_start=10
eval_freq=5
num_workers=32
experiment="AEKL"

runai submit \
  --name controlnet-aekl-v0 \
  --image aicregistry:5000/wds20:control_brain \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 16 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
  --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_aekl.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
