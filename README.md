# 运行环境
- python 3.8.5

- `sudo apt install libomp-dev`

- 安装mpi `sudo apt install mpich`

- 安装python库： `pip install -r requirements.txt`

- 根据cuda版本安装cupy：

  v11.1 (x86_64)  `pip install cupy-cuda111`

  v11.2 ~ 11.8 (x86_64 / aarch64) `pip install cupy-cuda11x`

  v12.x (x86_64 / aarch64) `pip install cupy-cuda12x`

# 参数目录结构：

--input_path

- val_B_labels_resized
- label_to_img.json

--img_path

  - imgs
  - labels

**测试过程需要在一张3090显卡上运行16个小时左右。**

