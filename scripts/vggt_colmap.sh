get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

dir="data/MipNeRF360"
post_fix="_vggt_opt_lr_5"

declare -a scenes=(
    "$dir$post_fix/bicycle"
    "$dir$post_fix/bonsai"
    "$dir$post_fix/counter"
    "$dir$post_fix/garden"
    "$dir$post_fix/kitchen"
    "$dir$post_fix/room"
    "$dir$post_fix/stump"
)

for scene_dir in $dir/*; do
# for scene_dir in "${scenes[@]}"; do
    while [ -d "$scene_dir" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running vggt on '$scene_dir'"
            CUDA_VISIBLE_DEVICES=$gpu_id python demo_colmap_ratio_opt.py \
                --scene_dir $scene_dir \
                --post_fix $post_fix \
                --shared_camera \
                --use_opt & \
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

output_dir=${dir}_vggt_opt
python utils/avg_metrics.py --output_dirs $dir$post_fix/* --save_path $dir$post_fix/vggt_results.txt
# python utils/avg_metrics.py --output_dirs $scenes --save_path $output_dir/vggt_results.txt