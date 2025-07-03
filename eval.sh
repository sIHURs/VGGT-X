get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

declare -a scenes=(
    "data/MipNeRF360/bonsai"
    "data/MipNeRF360/flowers"
    "data/MipNeRF360/stump"
)

dir="data/MipNeRF360"

for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python eval_pose_pcd.py --data_path $data_path &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait