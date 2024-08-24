# Art2Mus

<div align="center">

<img src="figures/logo.jpg" alt="Logo" width="350">

[![python](https://img.shields.io/badge/Python-3.10.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

# Introduction 🚀

Within this repository **Art2Mus**, an Artwork-Based Music Generation System, is proposed. Art2Mus leverages the AudioLDM2 architecture with an additional projection layer that enables digitized artworks to be used as conditioning information to guide the music generation process alongside text.

[ImageBind](https://github.com/facebookresearch/ImageBind) is used to generate image embeddings. The scripts used to compute the Fréchet Audio Distance (FAD) score during training are taken from [this repo](https://github.com/justivanr/frechet-audio-distance).

> [!NOTE]
> If you encounter any problems, we kindly ask you to open an issue summarizing the problem and, if possible, suggesting a solution if you have one.

## Installation of Requirements 🤖

> [!TIP]
> We **highly recommend using a virtual environment** for installing the required packages. If your Python version is not 3.10.12, consider using a **conda virtual environment**.

Before installing the requirements, update pip in your virtual environment to avoid potential installation issues. Use the following command to update pip:

```bash
python.exe -m pip install --upgrade pip
```

Once pip is updated, install the necessary libraries listed in the requirements.txt file to ensure the proper functioning of the model and the visualization of results. You can install them with the following command:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The final step may take some time due to the installation of various libraries.

## Run Art2Mus 🖼️🎵

> [!IMPORTANT]
> The code should work regardless of whether **CUDA** is installed on your machine. However, please note that inference will take longer on a CPU compared to a GPU.

Within the repository, you can find the following folders:

- [art2mus_/test_images/](test_images) ⇨ This folder contains digitized artworks for generating music. We have provided an [example digitized artwork](test_images/erin-hanson_thistles-on-orange-2016.jpg).
- [art2mus_/test_music/](test_music) ⇨ This folder contains the music generated based on the input artwork. An [example music file](test_music/art2mus_example.wav) is provided, which was generated based on the [example digitized artwork](test_images/erin-hanson_thistles-on-orange-2016.jpg).

To run Art2Mus with the example digitized artwork you will need to run the [test_art2mus.py](test_art2mus.py) script as it is. Alternatively, you can add your own digitized artwork to the [art2mus_/test_images/](test_images) folder and use it to generate new music! 

To generate music from your own digitized artwork, update the variable called **EXAMPLE_ARTWORK_PATH** variable in the [test_art2mus.py](test_art2mus.py) script with the path to your digitized artwork, then run the script.

## Train Art2Mus 🛠️

### Download Data 💾
---

The Artwork-Music dataset we used consists of **digitized artworks taken from the ArtGraph knowledge graph** and **music from the Large Free Music Archive (FMA) dataset**. 

ArtGraph's digitized artworks can be downloaded from [Zenodo](https://zenodo.org/records/8172374). Extract the .zip file contents, and place the imagesf2 folder in the [data/images/](data/images/) folder.

For the FMA dataset, you can easily download it using the [script](src/download_fma_dataset.py) we provide. As specified in the script, you need a Kaggle account to download the FMA dataset. The dataset will be stored in the [data/audio/](data/audio/) folder.

Additionally, you will need to download the digitized artworks and music embeddings. Click [here](https://drive.google.com/drive/folders/1rg4Q04ud0viLcLI9NEC7wtg5q5JgclRj?usp=sharing) to download them. After downloading, extract the safetensors (the .safetensors files) and place them in the **art2mus_/data/extra/** folder.

In the end, your data folder should look like this:

![plot](figures/example_data_folder.png?raw=true)

The **fma_large** folder will contain several subfolders (e.g., 001, 002, 003, etc.). The **imagesf2** folder should contain all the digitized artworks available in ArtGraph. Finally, the **extra** folder should contain the digitized artworks and music embeddings, as well as a [.json file](data/extra/image_audio_subset_df.json) that contains all the artwork-music pairs.

---

### Run Training 🎨🎶
---

> [!IMPORTANT]
> The following subfolders are required for the training to work properly: **art2mus_/src/art2mus/tmp_ground_truth** and **art2mus_/src/art2mus/tmp_generated**. If you do not find them under the **art2mus_/src/art2mus** folder, you need to create them. 

We provide both Art2Mus and Art2Mus-4 training codes to allow you to train your own version of Art2Mus. The training scripts can be found in the following folder: [art2mus_/src/art2mus](src/art2mus), and are named [art2mus_train.py](src/art2mus/art2mus_train.py) and [art2mus_4_train.py](src/art2mus/art2mus_4_train.py).

You can either train the Art2Mus (or Art2Mus-4) image projection layer from scratch or fine-tune it using the weights provided in the following folder: [art2mus_/art2mus_weights](art2mus_weights). If you want to train it from scratch, you need to move the weights associated with the model you want to train/tune out of [art2mus_/art2mus_weights](art2mus_weights).

> [!NOTE]
> Before launching the training script, we suggest carefully debugging the code to ensure you understand the overall training process. Additionally, we use Wandb to track our training, so you must [setup it](https://docs.wandb.ai/quickstart).

Below is an example of how you can launch the training/tuning using the accelerate library:

```bash
nohup accelerate launch src/art2mus/art2mus_train.py --num_epochs 20 --large_batch_size 8 --lr_warmup_steps 250  --dataloader_num_workers 16 --use_snr_gamma --set_wandb_offline
```

Details on each additional parameter that you can list after the training script can be found in the [train_test_argparse.py](src/art2mus/utils/train_test_argparse.py) script.


## Change Log
- 2024-08-18: Uploaded **weights** for Art2Mus and Art2Mus-4! 🌟
- 2024-08-24: Uploaded **training scripts** for Art2Mus and Art2Mus-4! 🌟

## TODO
- [x] Open-source Art2Mus's training code.
- [ ] Improve the quality of the generated music.
- [ ] Optimize the overall inference speed of Art2Mus.
- [ ] Test the impact of image transformations on the final generated music.