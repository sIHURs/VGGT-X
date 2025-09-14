
src_dir="data/MipNeRF360"
tgt_dir="data/MipNeRF360_vggt_opt_lr_5_avg_scale_500k_rand_order"

declare -a scenes=(
    "bicycle"
    "bonsai"
    "counter"
    "flowers"
    "garden"
    "kitchen"
    "room"
    "stump"
    "treehill"
)

for scene in "${scenes[@]}"; do
    python utils/clone_folder.py --src_dir $src_dir/$scene --tgt_dir $tgt_dir/$scene
done