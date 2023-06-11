output_dir="/project/outputs/samples/flair_to_t1w/"
stage1_path="/project/outputs/trained_models/autoencoder.pth"
diffusion_path="/project/outputs/trained_models/diffusion_model.pth"
controlnet_path="/project/outputs/trained_models/controlnet_model.pth"
stage1_config_file_path="/project/configs/stage1/aekl_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
controlnet_config_file_path="/project/configs/controlnet/controlnet_v0.yaml"
test_ids="/project/outputs/ids/test.tsv"
controlnet_scale=1.0
guidance_scale=7.0
x_size=20
y_size=28
scale_factor=0.3
num_workers=8
num_inference_steps=200

runai submit \
  --name  controlnet-sampling \
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
    python3 /project/src/python/testing/sample_flair_to_t1w.py \
      output_dir=${output_dir} \
      stage1_path=${stage1_path} \
      diffusion_path=${diffusion_path} \
      controlnet_path=${controlnet_path} \
      stage1_config_file_path=${stage1_config_file_path} \
      diffusion_config_file_path=${diffusion_config_file_path} \
      controlnet_config_file_path=${controlnet_config_file_path} \
      test_ids=${test_ids} \
      controlnet_scale=${controlnet_scale} \
      guidance_scale=${guidance_scale} \
      x_size=${x_size} \
      y_size=${y_size} \
      scale_factor=${scale_factor} \
      num_workers=${num_workers} \
      num_inference_steps=${num_inference_steps}
