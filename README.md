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

Within this repository **Art2Mus**, an Artwork-Based Music Generation System, is proposed. Art2Mus leverages the AudioLDM2 architecture with an additional projection layer that enables digitized artworks to be used as conditioning information to guide the music generation process alongside text.

[ImageBind](https://github.com/facebookresearch/ImageBind) is used to generate image embeddings.

> [!NOTE]
> If you encounter any problems, we kindly ask you to open an issue summarizing the problem and, if possible, suggesting a solution if you have one.

## Installation of Requirements

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

## Test Art2Mus

> [!IMPORTANT]
> The code should work regardless of whether **CUDA** is installed on your machine. However, please note that inference will take longer on a CPU compared to a GPU.

Within the repository, you can find the following folders:

- **art2mus\_/test_images/** ⇨ This folder contains digitized artworks for generating music. We have provided an [example digitized artwork](test_images/erin-hanson_thistles-on-orange-2016.jpg).
- **art2mus\_/test_music/** ⇨ This folder contains the music generated based on the input artwork. An [example music file](test_music/art2mus_example.wav) is provided, which was generated based on the [example digitized artwork](test_images/erin-hanson_thistles-on-orange-2016.jpg).

You can add your own digitized artwork to the **art2mus\_/test_images/** folder and use it to generate new music!

To generate new music, modify the path to the digitized artwork (there is a variable called **EXAMPLE_ARTWORK_PATH**) in the [test_art2mus.py](test_art2mus.py) script.

## Change Log

- 2024-08-18: Uploaded weights for Art2Mus and Art2Mus4! 🌟

## TODO

- [ ] Open-source Art2Mus's training code.
- [ ] Improve the quality of the generated music.
- [ ] Optimize the overall inference speed of Art2Mus.
- [ ] Test the impact of image transformations on the final generated music.
