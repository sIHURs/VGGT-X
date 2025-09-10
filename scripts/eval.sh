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

tgt_dir="data/MipNeRF360_vggt_opt_lr_fix_avg_scale_match_pcd_gamma_rand_order"
gt_dir="data/MipNeRF360"

for data_path in $tgt_dir/*; do
# for data_path in "${scenes[@]}"; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python utils/evaluate_pose_bin.py \
                --data_dir_pred $data_path \
                --data_dir_gt $gt_dir/$(basename $data_path) &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 15
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 15
        fi
    done
done
wait

python utils/gather_results.py $tgt_dir --format_float --result_filename pose_results.txt