# Are we pruning the correct channels in image-to-image translation models? （BIGP）
Yiyong Li1, Zhun Sun1*, Lichao2 *corresponding author, zhunsun@gmail.com
1 BIGO Ltd, 2 Tohoku University, 3 AIP, RIKEN

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Visualization Results
Image-to-image translation by (compressed) CycleGAN:
![](results.PNG)

### Installation
- Clone this repo:

  ```shell
    git clone https://github.com/liyiyong-nk/gan_BIGP.git
    cd gan_BIGP
  ```
- Install dependencies.

  ```shell
    conda create -n BIGP python=3.7.11
    conda activate BIGP
    pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 
    pip install -r requirements.txt 
  ```

## Training
### 1. Download dataset:
```
bash ./download_dataset <dataset_name>
```
This will download the dataset to folder `datasets/<dataset_name>` (e.g., `datasets/horse2zebra`).

### 2. Get the original dense CycleGAN:
#### summer2winter_yosemite dataset
Use the [official CycleGAN codes](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train original dense CycleGAN.
and load their parameter in our model

#### horse2zebra dataset
Use the [official CycleGAN codes](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train original dense CycleGAN.
and load their parameter in our model

### 3. Generate style transfer results on training set
Use the pretrained dense generator to generate style transfer results on training set and put the style transfer results to folder `train_set_result/<dataset_name>`.
For example, `train_set_result/horse2zebra/B/n02381460_2_fake.png` is the fake zebra image transferred from the real horse image `datasets/horse2zebra/train/A/n02381460_2.jpg` using the original dense CycleGAN.

### 4. Train and Compress
![](paramater_explanation.PNG)
code eg:
```
stage 1
python gan_bigp.py --lrw 8e-6 --alpha 1e-6 --contral_rate 0.001 --epochs 200 --dataset horse2zebra --task A2B --gpu 1
```
remark:
alpha*sparse_loss= 1e-6*334967.2812=0.3350 and perceptual=0.5922 should have a homologous quantity. The compression ratio you want to achieve may take several stages, which you should
follw the remark and modify the g_path in gan_bigp.py. The fuctions (BIG_loss, update_in) in gan_bigp.py can be flexibly applied to your project.

```
stage 2
python gan_bigp.py --lrw 8e-6 --alpha 9.8e-5 --contral_rate 0.0046 --epochs 200 --dataset horse2zebra --task A2B --gpu 1
```

```
stage 3
python gan_bigp.py --lrw 8e-6 --alpha 1.1e-4 --contral_rate 0.012 --epochs 200 --dataset horse2zebra --task A2B --gpu 1
```

The training results (checkpoints, loss curves, etc.) will be saved in `results/<dataset_name>/<task_name>`.

### 5. Extract compact subnetwork obtained by GS
We already updated the weight and bias of IN to 0. So it's easy to extract the subnet model.

```
python extract_subnet.py --dataset <dataset_name> --task <task_name> --model_str <model_str> 
```

The extracted subnetworks will be saved in `subnet_structures/<dataset_name>/<task_name>`

### 6. Testing
Given the (image_dir, result_dir, g_path), you can get the generated images.

```
python test.py 
```

The generated images will be saved in result_dir.


## Citation

If you use this code for your research, please cite our paper.
  ```shell
```

## Acknowledgements

Our code is developed based on https://github.com/VITA-Group/GAN-Slimming


# demo
