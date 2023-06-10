stage1_mlflow_path="/project/mlruns/837816334068618022/39336906f86c4cdc96fb6464b88c8c06/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/102676348294480761/a53f700f40184ff49f5f7e27fafece97/artifacts/final_model"
controlnet_mlflow_path="/project/mlruns/672765428205510835/92b0ea370a234caca38810246d4c60b7/artifacts/final_model"
output_dir="/project/outputs/trained_models/"

runai submit \
  --name controlnet-convert-model \
  --image aicregistry:5000/wds20:control_brain \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain_controlnet/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      stage1_mlflow_path=${stage1_mlflow_path} \
      diffusion_mlflow_path=${diffusion_mlflow_path} \
      controlnet_mlflow_path=${controlnet_mlflow_path} \
      output_dir=${output_dir}
