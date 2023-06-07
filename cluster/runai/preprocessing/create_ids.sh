output_dir="/project/outputs/ids/"

runai submit \
  --name create-ids \
  --image aicregistry:5000/wds20:control_brain \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 2 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
  --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/preprocessing/create_ids.py \
      output_dir=${output_dir}
