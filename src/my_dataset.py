import os
import sys
import tqdm
import json
import random
sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src/ImageBind")

import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


# ########################################################################################
# You must install safetensors in order for this script to work: pip install safetensors #
# ########################################################################################
from safetensors import safe_open

from art2mus.utils.imagebind_utils import load_model, gpu_if_available, check_cpu_or_gpu, generate_embeds


def store_dict_to_json(json_path, curr_dict):   
    """
    Store a dictionary to a JSON file.

    Params:
        - json_path (str): The file path where the JSON file will be saved.
        - curr_dict (dict): The dictionary to be saved to the JSON file.

    Returns:
        - None
    """      
    with open(json_path, 'w') as json_file:
        json.dump(curr_dict, json_file)
       

def read_dict_from_json(json_path):     
    """
    Read a dictionary from a JSON file.

    Params:
        - json_path (str): The file path of the JSON file to be read.

    Returns:
        - dict: The dictionary read from the JSON file.
    """       
    with open(json_path, 'r') as json_file:
        return json.load(json_file)


def remove_substring(string, data_type):
    """
    Removes a specified substring from the given string.

    Params:
        - string (str): The string from which to remove the substring.
        - data_type (str): Specifies the substring to remove. 
                           If 'image', removes 'imagesf2/' substring; 
                           if 'audio', removes 'fma-dataset-100k-music-wav-files/fma_large/' substring.

    Returns:
        - str: The string with a specified substring removed.
    """
    if data_type == 'image':
        substring = "imagesf2/"
    else:
        substring = "fma-dataset-100k-music-wav-files/fma_large/"
        
    return string.split(substring)[1]


def read_safetensor(file_path, data_type):
    """
    Reads tensors from a file and removes a specified substring from the keys.

    Params:
        - file_path (str): The path to the file containing tensors.
        - data_type (str): Specifies the substring to remove. 
                           If 'image', removes 'imagesf2/' substring; 
                           if 'audio', removes 'fma-dataset-100k-music-wav-files/' substring.

    Returns:
        - dict: A dictionary containing file paths and tensors.
    """
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    new_dict = {}

    for key, value in tensors.items():
        new_key = remove_substring(key, data_type)
        new_dict[new_key] = value
        
    return new_dict


def count_dict_values_occurrences(curr_dict):
    """
    Count the occurrences of values in a dictionary.

    Params:
        - curr_dict (dict): The dictionary whose values' occurrences need to be counted.

    Returns:
        - Counter: A Counter object with the counts of each value.
    """
    return Counter(curr_dict.values())


class ImageAudioDataset(Dataset):
    
    def __init__(self, json_file, images_dir, img_emb_file, audios_dir, audio_emb_file, transform=None):
        """
        Params:
            - json_file (string): path to the json file with all image-audio pairs
            - images_dir (string): directory containing all images
            - img_emb_file (string): path to the safetensor containing image-embedding pairs
            - audios_dir (string): directory containing all audios
            - audio_emb_file (string): path to the safetensor containing audio-embedding pairs
            - transform (callable, optional): optional transform to be applied on a sample
        """
        self.image_audio_pairs = pd.read_json(json_file)
        self.images_dir = images_dir
        self.img_embeds = read_safetensor(img_emb_file, 'image')
        self.audios_dir = audios_dir
        self.audio_embeds = read_safetensor(audio_emb_file, 'audio')
        self.transform = transform

    def __len__(self):
        """
        Return the number of image-audio pairs in the dataset.

        Params:
            - None

        Returns:
            - int: The total number of image-audio pairs in the dataset.
        """
        return len(self.image_audio_pairs)
    
    def __get_image_audio_pair__(self, idx):
        """
        Retrieve the image-audio pair at the given index.

        Params:
            - idx (int): Index of the desired image-audio pair.

        Returns:
            - tuple: A tuple containing the paths to the image and audio files.
        """
        pair = self.image_audio_pairs.iloc[idx]
        image_name = self.images_dir + pair['Image']
        audio_name = self.audios_dir + remove_substring(pair['Audio'], 'audio')

        return image_name, audio_name
    
    def __getitem__(self, idx, full_log=False):
        """
        Retrieve the image embedding and audio at the given index.

        Params:
            - idx (int): Index of the desired image-audio pair.
            - audio_spectrogram (bool, optional): If True, convert the audio to its spectrogram. 
                                                  Default value: False.
            - full_log (bool, optional): If True, print detailed logs. 
                                         Default value: False.

        Returns:
            - tuple: A tuple containing the image embedding and the audio path.

        """
        # Retrieve the image-audio pair from
        pair = self.image_audio_pairs.iloc[idx]
        image_name = pair['Image']
        audio = self.audios_dir + remove_substring(pair['Audio'], 'audio')
        
        # Retrieve image embedding from the images embeddings' safetensor.
        # Use the image embedding if found, else compute it using ImageBind.
        if self.img_embeds[image_name].numel() != 0:
            if full_log:
                print("Found embedding!")
            image_emb = self.img_embeds[image_name]
        else:
            if full_log:
                print("Computing embedding!")
            imagebind = load_model(False)
            gpu_if_available(imagebind, full_log=False)
            device = check_cpu_or_gpu(imagebind, full_log=False)
            img = [self.images_dir + image_name]
            image_emb = generate_embeds(imagebind, device, img, extract_emb=True, emb_type='vision')
            
        if full_log:
            print(f"Image: {image_name}\nImage Embedding: {image_emb}\nAudio: {audio}")

        """
        TODO: Introduce image transformations that can be found in this paper:
        - Vis2Mus => https://arxiv.org/pdf/2211.05543
        Why? Check if the output audio changes when applying transformations to the input image.
        
        if self.transform:                      
            image = self.transform(image)
        """

        return image_emb, audio
    
    def train_val_test_split(self, val_size=0.2, test_size=None, random_state=42):
        """
        Splits the dataset into training, validation, and optionally test sets.
        
        Params:
            - val_size: Proportion of the dataset to include in the validation split. 
                        Default value: 0.2.
            - test_size: Proportion of the dataset to include in the test split. 
                         If None, only the training and validation sets will be created. 
                         Default value: None.
            - random_state: Controls the shuffling applied to the data before splitting. 
                            Default value: 42.
            
        Returns:
            - train_dataset: The training subset of the dataset.
            - val_dataset: The validation subset of the dataset.
            - test_dataset (Optional): The test subset of the dataset. Only returned if `test_size` is provided.
        """
        indices = np.arange(len(self.image_audio_pairs))
        
        if test_size:
            # Calculate the split sizes
            train_val_size = 1.0 - test_size
            # relative_val_size = val_size / train_val_size
            
            # First split into train+validation and test
            train_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Further split train+validation into train and validation
            test_indices, val_indices = train_test_split(
                test_indices,
                test_size=100,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            test_dataset = Subset(self, test_indices)
            
            return train_dataset, val_dataset, test_dataset
        
        else:
            # Split into train and validation only
            train_indices, test_indices = train_test_split(
                indices,
                test_size=val_size,
                random_state=random_state
            )
            
            # Determine the test size based on the desired number of elements in val_indices
            """ ---- We decided to work with only 100 validation instances ---- """
            desired_val_size = 100
            total_indices = len(test_indices)
            test_size = max(0, total_indices - desired_val_size) / total_indices
            
            val_indices, test_indices = train_test_split(
                test_indices,
                test_size=test_size,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            
            return train_dataset, val_dataset
        
    def audio_path_from_img_emb(self, curr_img_emb):
        """
        Retrieve the audio path corresponding to a given image embedding.

        Params:
            - curr_img_emb (str): The embedding of the image for which the audio path needs to be retrieved.

        Returns:
            - str or None: The audio path corresponding to the given image embedding. If no matching
            image embedding is found, returns None.
        """
        for key, tensor in self.img_embeds.items():
            if torch.allclose(tensor, curr_img_emb):
                img_name = key
                break
            
        audio_path = self.image_audio_pairs[self.image_audio_pairs['Image'] == img_name]['Audio'].iloc[0]
        audio_path = self.audios_dir + remove_substring(audio_path, 'audio')
        return audio_path
    

    def __get_aud_emb_from_path__(self, aud_path):
        """
        Retrieve the audio embedding corresponding to a given audio path.

        Params:
            - aud_path (str): The path to a specific audio whose embedding needs to be retrieved.

         Returns:
            - torch.Tensor: The audio embedding tensor corresponding to the given audio path. If no matching
            audio embedding is found, raises a KeyError.
        """        
        split_string = "fma_large/"
        if split_string in aud_path:
            aud_path = aud_path.split(split_string)[1]

        return self.audio_embeds[aud_path]
    
    
    def __get_image_name_from_emb__(self, curr_img_emb):
        """
        Retrieve the image path corresponding to a given image embedding.

        Params:
            - curr_img_emb (str): The embedding of the image for which the audio path needs to be retrieved.

        Returns:
            - str or None: The image path corresponding to the given image embedding. If no matching
            image embedding is found, returns None.
        """        
        for key, tensor in self.img_embeds.items():
            if torch.allclose(tensor, curr_img_emb):
                img_name = key
                break
            
        img_name = self.images_dir + img_name
        return img_name
    

class CombinedDatasets(ImageAudioDataset):
    
    def __init__(self, historical_json_file, emotional_json_file, create_mixed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.historical_data = pd.read_json(historical_json_file)
        self.emotional_data = pd.read_json(emotional_json_file)
        self.sampled_emotional_data = self.__random_sample_emotions__()
        self.data_dict = {}
        if create_mixed:
            self.__compute_df_dicts__()
            self.mixed_dataset = self.__mix_datasets__()
    
    def __mix_datasets__(self):
        """
        Mix datasets by randomly selecting an audio file for each artwork based on the available datasets.

        Returns:
            - pandas.DataFrame: A DataFrame containing two columns: "Image" with artwork names 
                                and "Audio" with randomly sampled audio paths.
        """
        print(f"=========================================================================================================\n"
              f"Mixing datasets.. 🤗")
        artworks = list(self.image_audio_pairs['Image'])
        audios = []
        
        self.art_data_dict = read_dict_from_json(conf.EXTRA_DATA_DIR + "art_datasets_dict.json")
        
        # Sample a random audio for each artwork based on the no. its emotions
        for artw in tqdm.tqdm(artworks, desc="Processing artworks"):
            
            # Retrieve datasets containing current artwork
            datasets = self.art_data_dict[artw]
            # Randomly choose a dataset among the available ones
            rnd_dataset = self.__choose_random_dataset__(datasets_list=datasets)
            # Add artwork-dataset to data_dict to keep track of the chosen datasets
            self.data_dict[artw] = rnd_dataset
            # Append associated audio to the audios list
            audios.append(self.__retrieve_artw_audio_dataset__(img_to_find=artw, dataset_type=rnd_dataset))
        
        print(f"Datasets mixed! ✅\n"
              f"=========================================================================================================\n"
              f"No. audios taken from each dataset: {count_dict_values_occurrences(self.data_dict)} ⛳\n"
              f"========================================================================================================="
              )
            
        return pd.DataFrame({"Image": artworks, "Audio": audios})
    
    
    def __compute_df_dicts__(self):
        """
        Compute dictionaries for faster lookup of audio paths associated with artworks from different datasets.

        Returns:
            None
        """
        self.basic_audio_dict = self.image_audio_pairs.set_index('Image')['Audio'].to_dict()
        self.historical_audio_dict = self.historical_data.set_index('Image')['Audio'].to_dict()
        self.emotional_audio_dict = self.sampled_emotional_data.set_index('Image')['Audio'].to_dict()
    
    
    def __retrieve_artw_audio_dataset__(self, img_to_find, dataset_type="basic"):
        """
        Retrieve the audio associated with an artwork from the specified dataset.

        Params:
            - img_to_find (str): The identifier of the image to find.
            - dataset_type (str, optional): The type of dataset to retrieve from. Options are "basic", "historical", or "emotional".
                                            Default value: "basic".

        Returns:
            - str: The path to the audio file associated with the image.
        """
        if dataset_type == "historical":
            dataset_to_use = self.historical_audio_dict
        elif dataset_type == "emotional":
            dataset_to_use = self.emotional_audio_dict
        else:
            dataset_to_use = self.basic_audio_dict

        img_aud = dataset_to_use.get(img_to_find, None)
        if img_aud is not None:
            return img_aud
        else:
            return None
        

    def __random_sample_emotions__(self):
        """
        Generate a DataFrame with randomly sampled audio paths corresponding to each artwork in the emotional dataset.
        Sample a random audio for each artwork based on the number of emotions associated with it.

        Returns:
            - pandas.DataFrame: A DataFrame containing two columns: "Image" with artwork names 
                                and "Audio" with randomly sampled audio paths.
        """
        artworks = list(self.emotional_data.columns)
        audios = []
        
        # Sample a random audio for each artwork based on the no. its emotions
        for artw in artworks:
            artw_audios = self.emotional_data[artw].dropna()
            rand_id = random.randint(0, len(artw_audios) - 1)
            audios.append(artw_audios.iloc[rand_id])
            
        return pd.DataFrame({"Image": artworks, "Audio": audios})
        
        
    def __get_artw_dataset_from_emb__(self, img_emb=None):
        """
        Retrieve the artwork dataset based on the given image embedding.

        Params:
            - img_emb: The image embedding from which to extract the artwork name.

        Returns:
            - The dataset corresponding to the extracted artwork name from the data dictionary.
        """
        artw_name = self.__get_image_name_from_emb__(img_emb)
        artw_name = artw_name.split('imagesf2/')[1]
        
        return self.data_dict[artw_name].upper()

        
        
    def __len__(self, dataset_type="basic"):
        """
        Return the number of image-audio pairs in the specified dataset.

        Params:
            - dataset_type (str): The type of dataset to evaluate. Options are "basic", "historical", "mixed", or "emotional".
                                  Default value: "basic".

        Returns:
            - int: The total number of image-audio pairs in the specified dataset.
                   If an unknown dataset_type is provided, returns a tuple with counts from all datasets.
        """
        if dataset_type == "basic":
            return len(self.image_audio_pairs)
        elif dataset_type == "historical":
            return len(self.historical_data)
        elif dataset_type == "emotional":
            return len(self.sampled_emotional_data)
        elif dataset_type == "mixed":
            return len(self.mixed_dataset)
        else:
            return len(self.image_audio_pairs), len(self.historical_data), len(self.sampled_emotional_data), len(self.mixed_dataset)
    
    
    def __get_image_audio_pair__(self, idx, dataset_type="basic"):
        """
        Retrieve the image-audio pair at the given index from the specified dataset.

        Params:
            - idx (int): Index of the desired image-audio pair.
            - dataset_type (str): The type of dataset to retrieve from. Options are "basic", "historical", "mixed", or "emotional".
                                  Default value: "basic".

        Returns:
            - tuple: A tuple containing the paths to the image and audio files.
        """
        if dataset_type == "historical":
            dataset_to_use = self.historical_data
        elif dataset_type == "emotional":
            dataset_to_use = self.sampled_emotional_data
        elif dataset_type == "mixed":
            dataset_to_use = self.mixed_dataset
        else:
            dataset_to_use = self.image_audio_pairs

        pair = dataset_to_use.iloc[idx]
        image_name = self.images_dir + pair['Image']
        audio_name = self.audios_dir + remove_substring(pair['Audio'], 'audio')

        return image_name, audio_name
   
    
    def __getitem__(self, idx, full_log=False, dataset_type="basic"):
        """
        Retrieve the image embedding and audio at the given index from the specified dataset.

        Params:
            - idx (int): Index of the desired image-audio pair.
            - full_log (bool, optional): If True, print detailed logs. Default value: False.
            - dataset_type (str): The type of dataset to retrieve from. Options are "basic", "historical", "mixed", or "emotional".
                                  Default value: "basic".

        Returns:
            - tuple: A tuple containing the image embedding and the audio path.
        """
        # Retrieve the image-audio pair from
        
        if dataset_type == "historical":
            dataset_to_use = self.historical_data
        elif dataset_type == "emotional":
            dataset_to_use = self.sampled_emotional_data
        elif dataset_type == "mixed":
            dataset_to_use = self.mixed_dataset
        else:
            dataset_to_use = self.image_audio_pairs

        pair = dataset_to_use.iloc[idx]
        image_name = pair['Image']
        audio = self.audios_dir + remove_substring(pair['Audio'], 'audio')
        
        # Retrieve image embedding from the images embeddings' safetensor.
        # Use the image embedding if found, else compute it using ImageBind.
        if self.img_embeds[image_name].numel() != 0:
            if full_log:
                print("Found embedding!")
            image_emb = self.img_embeds[image_name]
        else:
            if full_log:
                print("Computing embedding!")
            imagebind = load_model(False)
            gpu_if_available(imagebind, full_log=False)
            device = check_cpu_or_gpu(imagebind, full_log=False)
            img = [self.images_dir + image_name]
            image_emb = generate_embeds(imagebind, device, img, extract_emb=True, emb_type='vision')
            
        if full_log:
            print(f"Image: {image_name}\nImage Embedding: {image_emb}\nAudio: {audio}")

        return image_emb, audio
 
        
    def train_val_test_split(self, val_size=0.2, test_size=None, random_state=42, dataset_type="basic"):
        """
        Splits the dataset into training, validation, and optionally test sets.
        
        Params:
            - val_size: Proportion of the dataset to include in the validation split. 
                        Default value: 0.2.
            - test_size: Proportion of the dataset to include in the test split. 
                        If None, only the training and validation sets will be created. 
                        Default value: None.
            - random_state: Controls the shuffling applied to the data before splitting. 
                            Default value: 42.
            - dataset_type (str): The type of dataset to split. Options are "basic", "historical", "mixed" or "emotional".
            
        Returns:
            - train_dataset: The training subset of the dataset.
            - val_dataset: The validation subset of the dataset.
            - test_dataset (Optional): The test subset of the dataset. Only returned if `test_size` is provided.
        """
        if dataset_type=="emotional":
            indices = np.arange(len(self.sampled_emotional_data))
        elif dataset_type=="historical":
            indices = np.arange(len(self.historical_data))
        elif dataset_type == "mixed":
            indices = np.arange(len(self.mixed_dataset))
        else:
            indices = np.arange(len(self.image_audio_pairs))
            
        if test_size:
            # Calculate the split sizes
            train_val_size = 1.0 - test_size
            relative_val_size = val_size / train_val_size
            
            # First split into train+validation and test
            train_val_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Further split train+validation into train and validation
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=relative_val_size,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            test_dataset = Subset(self, test_indices)
            
            return train_dataset, val_dataset, test_dataset
        
        else:
            # Split into train and validation only
            train_indices, test_indices = train_test_split(
                indices,
                test_size=val_size,
                random_state=random_state
            )
            
            # Determine the test size based on the desired number of elements in val_indices
            """ ---- We decided to work with only 100 validation instances ---- """
            desired_val_size = 100
            total_indices = len(test_indices)
            test_size = max(0, total_indices - desired_val_size) / total_indices
            
            val_indices, test_indices = train_test_split(
                test_indices,
                test_size=test_size,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            
            return train_dataset, val_dataset
     
        
    def audio_path_from_img_emb(self, curr_img_emb, dataset_type="basic"):
        """
        Retrieve the audio path corresponding to a given image embedding from the specified dataset.

        Params:
            - curr_img_emb (str): The embedding of the image for which the audio path needs to be retrieved.
            - dataset_type (str): The type of dataset to retrieve from. Options are "basic", "historical", "mixed", or "emotional".
                                  Default value: "basic".

        Returns:
            - str or None: The audio path corresponding to the given image embedding. If no matching
                        image embedding is found, returns None. If dataset_type is "emotional", also prints
                        the associated emotion of the audio.
        """
        for key, tensor in self.img_embeds.items():
            if torch.allclose(tensor, curr_img_emb):
                img_name = key
                break
            
        if dataset_type == "historical":
            dataset_to_use = self.historical_data
        elif dataset_type == "emotional":
            dataset_to_use = self.sampled_emotional_data
        elif dataset_type == "mixed":
            dataset_to_use = self.mixed_dataset
        else:
            dataset_to_use = self.image_audio_pairs
            
        audio_path = dataset_to_use[dataset_to_use['Image'] == img_name]['Audio'].iloc[0]
        audio_path = self.audios_dir + remove_substring(audio_path, 'audio')
            
        return audio_path
  
    
    def __datasets_containg_image__(self, img_emb=None, img_path=None):
        """
        Retrieve the datasets containing an image based on either its embedding or path.

        Params:
            - img_emb (str, optional): The embedding of the image.
            - img_path (str, optional): The path of the image.
            
        Returns:
            - list: List of dataset names containing the image.
        """
        if img_emb is not None and img_path is None:
            img_name = self.__get_image_name_from_emb__(img_emb).split('imagesf2/')[1]
        else:
            if 'imagesf2/' in img_path:
                img_name = img_path.split('imagesf2/')[1]
            else:
                img_name = img_path
        
        datasets_containing_img = []

        image_audio_pairs_set = set(self.image_audio_pairs['Image'])
        historical_data_pairs_set = set(self.historical_data['Image'])
        emotional_pairs_set = set(self.sampled_emotional_data['Image'])
        
        if img_name in image_audio_pairs_set:
            datasets_containing_img.append('basic')

        if img_name in historical_data_pairs_set:
            datasets_containing_img.append('historical')
            
        if img_name in emotional_pairs_set:
            datasets_containing_img.append('emotional')
        
        return datasets_containing_img
    
    
    def __choose_random_dataset__(self, datasets_list=None):
        """
        Choose a random dataset from the provided list.

        Params:
            - datasets_list (list, optional): List of dataset names.

        Returns:
            - str: Randomly selected dataset name.
        """
        if datasets_list is None:
            return "basic"
        else:
            return random.choice(datasets_list)
        
        
class DescAudioDataset(ImageAudioDataset):
    
    def __init__(self, artw_artwdesc_file, artwork_desc_music_file, *args, **kwargs):
        super().__init__(*args, **kwargs)       
        self.artw_artwdesc = pd.read_json(artw_artwdesc_file)
        self.artwork_desc_music = pd.read_json(artwork_desc_music_file)

    
    def __get_image_path_from_description__(self, artw_desc=None):
        """
        Retrieve the artwork digitized image path based on its description.

        Params:
            - artw_desc: The artwork description from which to extract the artwork digitized image path.

        Returns:
            - The digitized image path.
        """
        return self.images_dir + self.artw_artwdesc[self.artw_artwdesc['ArtwDesc'] == artw_desc]['Image'].iloc[0]


    def __get_image_emb_from_desc__(self, artw_desc=None):
        """
        Retrieve the artwork digitized image embedding based on its description.

        Params:
            - artw_desc: The artwork description from which to extract the artwork digitized image path.

        Returns:
            - The digitized image embedding.
        """
        img_path = self.__get_image_path_from_description__(artw_desc)
        return self.img_embeds[img_path.split('imagesf2/')[1]]

        
    def __len__(self):
        """
        Return the number of artwdesc-audio pairs in the dataset.
        
        Params:
            - None

        Returns:
            - int: The total number of artwdesc-audio pairs in the dataset.
        """
        return len(self.artwork_desc_music)

    
    def __get_image_audio_pair__(self, idx):
        """
        Retrieve the artwdesc-audio pair at the given index.

        Params:
            - idx (int): Index of the desired artwdesc-audio pair.

        Returns:
            - tuple: A tuple containing the paths to the image and audio files.
        """
        pair = self.artwork_desc_music.iloc[idx]
        image_name = self.images_dir + self.__get_image_path_from_description__(pair['ArtwDesc'])
        audio_name = self.audios_dir + remove_substring(pair['Audio'], 'audio')

        return image_name, audio_name
   
    
    def __getitem__(self, idx, full_log=False):
        """
        Retrieve the image embedding and audio at the given index from the specified dataset.

        Params:
            - idx (int): Index of the desired image-audio pair.
            - full_log (bool, optional): If True, print detailed logs. Default value: False.
          
        Returns:
            - tuple: A tuple containing the artwork description and the audio path.
        """
        # Retrieve the image-audio pair from

        pair = self.artwork_desc_music.iloc[idx]
        artw_desc = pair['ArtwDesc']
        audio = self.audios_dir + remove_substring(pair['Audio'], 'audio')
            
        if full_log:
            print(f"Artwork description: {artw_desc}\n\nAudio: {audio}")

        return artw_desc, audio
 
        
    def train_val_test_split(self, val_size=0.2, test_size=None, random_state=42):
        """
        Splits the dataset into training, validation, and optionally test sets.
        
        Params:
            - val_size: Proportion of the dataset to include in the validation split. 
                        Default value: 0.2.
            - test_size: Proportion of the dataset to include in the test split. 
                        If None, only the training and validation sets will be created. 
                        Default value: None.
            - random_state: Controls the shuffling applied to the data before splitting. 
                            Default value: 42.
            - dataset_type (str): The type of dataset to split. Options are "basic", "historical", "mixed" or "emotional".
            
        Returns:
            - train_dataset: The training subset of the dataset.
            - val_dataset: The validation subset of the dataset.
            - test_dataset (Optional): The test subset of the dataset. Only returned if `test_size` is provided.
        """
        indices = np.arange(len(self.artwork_desc_music))
            
        if test_size:
            
            # First split into train+validation and test
            train_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Further split train+validation into train and validation
            test_indices, val_indices = train_test_split(
                test_indices,
                test_size=100,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            test_dataset = Subset(self, test_indices)
            
            return train_dataset, val_dataset, test_dataset
        
        else:
            # Split into train and validation only
            train_indices, test_indices = train_test_split(
                indices,
                test_size=val_size,
                random_state=random_state
            )
            
            # Determine the test size based on the desired number of elements in val_indices
            """ ---- We decided to work with only 100 validation instances ---- """
            desired_val_size = 100
            total_indices = len(test_indices)
            test_size = max(0, total_indices - desired_val_size) / total_indices
            
            val_indices, test_indices = train_test_split(
                test_indices,
                test_size=test_size,
                random_state=random_state
            )
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            
            return train_dataset, val_dataset