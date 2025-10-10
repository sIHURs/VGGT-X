<p align="center">
  <h1 align="center"><strong> <img src="assets/72.png" width="30" height="30">  VGGT-X: When VGGT Meets Dense Novel View Synthesis</strong></h1>

  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; Linketic</em>
  </p>

</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.25191-b31b1b.svg)](http://arxiv.org/abs/2509.25191)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://dekuliutesla.github.io/vggt-x.github.io/)

</div>

<div align="center">
    <img src="assets/teaser.png">
</div>


## 📰 News
**[2025.09.30]** arXiv release of our VGGT-X!


## 💡 Overview

<div align="center">
    <img src='assets/pipeline.png'/>
</div>

<b>VGGT-X</b> takes dense multi-view images as input. It first uses memory-efficient VGGT to losslessly predict 3D key attributes, which we name as VGGT--. Then, a fast global alignment module refines the predicted camera poses and point clouds. Finally, a robust joint pose and 3DGS training pipeline is applied to produce high-fidelity novel view synthesis.

## ⚡ Quick Start

First, clone this repository to your local machine, and install the dependencies. 

```bash
git clone https://github.com/Linketic/VGGT-X.git 
cd VGGT-X
conda create -n vggt_x python=3.10
conda activate vggt_x
pip install -r requirements.txt
```

Now, put the image collection to path/to/your/scene/images. Please ensure that the images are stored in `/YOUR/SCENE_DIR/images/`. This folder should contain only the images. Then run the model and get COLMAP results:

```bash
python demo_colmap.py --scene_dir path/to/your/scene --shared_camera --use_opt
```

The reconstruction result (camera parameters and 3D points) will be automatically saved under `/YOUR/SCENE_DIR_vggt_x/sparse/` in the COLMAP format (currently only supports `PINHOLE` camera type), such as:

``` 
SCENE_DIR/
  └── images/
SCENE_DIR_vggt_x/
  ├── images/
  └── sparse/
      ├── cameras.bin
      ├── images.bin
      └── points3D.bin
```
Note that it would soft link everything in `/YOUR/SCENE_DIR/` to the new folder `/YOUR/SCENE_DIR_vggt_x/`, except for the `sparse/` folder. It minimizes additional storage usage and facilitates usage of reconstruction results.

## 🔍 Detailed Usage

<details>
<summary>Click to expand</summary>

  #### --post_fix
  Post fix for the output folder (`_vggt_x` by default). You can set any desired name for the output folder.
  #### --seed
  Random seed for reproducibility.
  #### --use_ga
  If specified, the global alignment will be applied to VGGT output for better reconstruction. The matching results would be saved to `/YOUR/SCENE_DIR_vggt_x/matches.pt`.
  #### --save_depth
  If specified, it would save the depth and confidence to `/YOUR/SCENE_DIR_vggt_x/estimated_depths/` and `/YOUR/SCENE_DIR_vggt_x/estimated_confs/` as .npy files.
  #### --total_frame_num
  If specified, it would use first the `total_frame_num` images for reconstruction. Otherwise, all images will be considered in processing.
  #### --chunk_size
  Chunk size for frame-wise operation in VGGT. Default value is 512. **You can specify a smaller value to release VGGT computation burden**.
  #### --max_query_pts
  Maximum query points for XFeat matching. For each pair, XFeat would generate `max_query_pts` matches. If not specified, it is set to 4096 if number of images is less than 500 and 2048 otherwise. **You can specify a smaller value to release GA computation burden**.
  #### --max_points_for_colmap
  Maximum number for colmap point cloud. Default value is 500000.
  #### --shared_camera
  If specified, it would use shared camera for all images.
</details>


## Interactive Demo

We provide multiple ways to visualize your 3D reconstructions. Before using these visualization tools, install the required dependencies:

```bash
pip install -r requirements_demo.txt
```

### Interactive 3D Visualization

**Please note:** VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, independent of VGGT's processing time. The visualization is slow especially when the number of images is large.


#### Gradio Web Interface

Our Gradio-based interface allows you to upload images/videos, run reconstruction, and interactively explore the 3D scene in your browser. You can launch this in your local machine or try it on [Hugging Face](https://huggingface.co/spaces/facebook/vggt).


```bash
python demo_gradio.py
```

<details>
<summary>Click to preview the Gradio interactive interface</summary>

![Gradio Web Interface Preview](https://jytime.github.io/data/vggt_hf_demo_screen.png)
</details>


#### Viser 3D Viewer

Run the following command to run reconstruction and visualize the point clouds in viser. Note this script requires a path to a folder containing images. It assumes only image files under the folder. You can set `--use_point_map` to use the point cloud from the point map branch, instead of the depth-based point cloud.

```bash
python demo_viser.py --image_folder path/to/your/images/folder
```

## Integration with Gaussian Splatting


The exported COLMAP files can be directly used with [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian Splatting training. Install `gsplat` following their official instructions (we recommend `gsplat==1.3.0`):

An example command to train the model is:
```
cd gsplat
python examples/simple_trainer.py  default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```


## Runtime and GPU Memory

We benchmark the runtime and GPU memory usage of VGGT's aggregator on a single NVIDIA H100 GPU across various input sizes. 

| **Input Frames** | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|:----------------:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:---:|:---:|
| **Time (s)**     | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| **Memory (GB)**  | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

Note that these results were obtained using Flash Attention 3, which is faster than the default Flash Attention 2 implementation while maintaining almost the same memory usage. Feel free to compile Flash Attention 3 from source to get better performance.


## Acknowledgements

Thanks to these great repositories: [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion), [VGGSfM](https://github.com/facebookresearch/vggsfm), [CoTracker](https://github.com/facebookresearch/co-tracker), [DINOv2](https://github.com/facebookresearch/dinov2), [Dust3r](https://github.com/naver/dust3r), [Moge](https://github.com/microsoft/moge), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D](https://github.com/YvanYin/Metric3D) and many other inspiring works in the community.

## Checklist

- [x] Release the training code
- [ ] Release VGGT-500M and VGGT-200M


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

## 🤗 Citation
If you find this repository useful for your research, please use the following BibTeX entry for citation.

    @misc{liu2025vggtxvggtmeetsdense,
          title={VGGT-X: When VGGT Meets Dense Novel View Synthesis}, 
          author={Yang Liu and Chuanchen Luo and Zimo Tang and Junran Peng and Zhaoxiang Zhang},
          year={2025},
          eprint={2509.25191},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2509.25191}, 
    }
