get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

dir="data/MipNeRF360"  # replace with your dataset path
post_fix="_vggt_x"  # replace with your desired postfix for output directories

# List of scene directories if you want to run specific scenes
declare -a scenes=(
    "$dir$post_fix/bicycle"
    "$dir$post_fix/bonsai"
    "$dir$post_fix/counter"
    "$dir$post_fix/garden"
    "$dir$post_fix/kitchen"
    "$dir$post_fix/room"
    "$dir$post_fix/stump"
)

# Run all scenes in the dataset directory by default
for scene_dir in $dir/*; do
# for scene_dir in "${scenes[@]}"; do
    while [ -d "$scene_dir" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running vggt-x on '$scene_dir'"
            CUDA_VISIBLE_DEVICES=$gpu_id python demo_colmap.py \
                --scene_dir $scene_dir \
                --post_fix $post_fix \
                --shared_camera \
                --use_ga & \
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

python utils/gather_results.py $dir$post_fix --format_float --result_filename eval_results.txt
