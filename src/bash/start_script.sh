#!/usr/bin/bash
#print user info
echo "$(id)"

# Define mlflow
export MLFLOW_TRACKING_URI=file:/project/mlruns
echo ${MLFLOW_TRACKING_URI}

# Define place to save lpips pretrained models
export TORCH_HOME=/project/outputs/torch_home
export HF_HOME=/project/outputs/hf_home

# parse arguments
CMD=""
for i in $@; do
  if [[ $i == *"="* ]]; then
    ARG=${i//=/ }
    CMD=$CMD"--$ARG "
  else
    CMD=$CMD"$i "
  fi
done

# execute comand
echo $CMD
$CMD
