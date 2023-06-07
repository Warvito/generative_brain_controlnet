for i in {0..39};do
  start=$((i*1000))
  stop=$(((i+1)*1000))

  runai submit \
    --name create-dataset-${start}-${stop} \
    --image aicregistry:5000/wds20:control_brain \
    --backoff-limit 0 \
    --gpu 0 \
    --cpu 2 \
    --large-shm \
    --run-as-user \
    --host-ipc \
    --project wds20 \
    --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
    --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/source_data/ \
    --volume /nfs/home/wds20/datasets/Biobank/derivatives/2d_controlnet/:/data/ \
    --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/preprocessing/create_png_dataset.py \
        start=${start} \
        stop=${stop}
    sleep 2
done
