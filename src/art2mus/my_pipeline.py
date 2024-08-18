# Copyright 2024 CVSSP, ByteDance and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import scipy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src/art2mus")
sys.path.append(PROJ_DIR + "/src/ImageBind")

# AudioLDM2's stuff
from my_modeling import MyAudioLDM2ProjectionModel, MyAudioLDM2UNet2DConditionModel
from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModelOutput
from my_modeling import add_special_tokens

# ImageBind's stuff
from art2mus.utils.imagebind_utils import load_model, generate_embeds

import torch.nn as nn

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline

from sklearn.metrics.pairwise import cosine_similarity

if is_librosa_available():
    import librosa

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TEMP_AUDIO_DIR = PROJ_DIR + "/src/tmp_audios/"
IMG_PROJ_LAYER_WEIGHTS = PROJ_DIR + "/art2mus_weights/img_proj_layer.pt"

EMBEDS_DTYPE = torch.float16
ATT_MASK_DTYPE = torch.int64

BASIC_DATASET_TEXT = "Music representing the content of this artwork"
HISTORICAL_DATASET_TEXT = "Music representing the period and place of this artwork"
EMOTIONAL_DATASET_TEXT = "Music that evokes the emotion () of this artwork"

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
"""


def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


def print_weights(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")


class ImageProjectionLayer(ModelMixin, ConfigMixin):
    """
    A projection layer that projects image embeddings into a higher-dimensional space.
    It consists of two linear projection layers with a Tanh activation in between.

    Attributes:
        - language_model_dim (int): The dimensionality of the T5 language model embeddings.
        - image_emb_dim (int): The dimensionality of ImageBind's embeddings.
        - gpt_model_dim (int): The dimensionality of GPT2's embeddings.
        - projection1 (nn.Linear): The first linear projection layer.
        - projection2 (nn.Linear): The second linear projection layer.
        - act1 (nn.Tanh): The Tanh activation function.
        - first_mult (int): The multiplication factor for the first projection layer's output dimension.
        - second_mult (int): The multiplication factor for the second projection layer's output dimension.
        
        Currently, both language_model_dim and second_mult are not employed in anything.
        We expect to utilize them in the future.
    """
    @register_to_config
    def __init__(self):
        super().__init__()
        """ T5's emb dim: 1024. """
        self.language_model_dim = 1024
        """ ImageBind's emb dim: 1024. """
        self.image_emb_dim = 1024
        """ GPT2's emb dim: 768. """
        self.gpt_model_dim = 768
        
        self.first_mult = 2
        self.second_mult = 4 
        
        # Prev 4 and 8
        # Prev 16 and 32
        self.projection1 = nn.Linear(self.image_emb_dim, self.first_mult * self.gpt_model_dim)
        self.act1 = nn.Tanh()
        self.projection2 = nn.Linear(self.first_mult * self.gpt_model_dim, self.gpt_model_dim)
        # self.projection2 = nn.Linear(self.first_mult * self.gpt_model_dim, self.second_mult * self.gpt_model_dim)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):
        """
        Forward pass. The input hidden states are projected through two linear layers
        with a Tanh activation in between.

        Params:
            - hidden_states (Optional[torch.FloatTensor]): The input tensor representing the image embeddings.

        Returns:
            - AudioLDM2ProjectionModelOutput: An object containing the projected hidden states.
        """
        hidden_states = self.projection1(hidden_states)
        hidden_states = self.act1(hidden_states)
        hidden_states = self.projection2(hidden_states)
        # hidden_states = hidden_states.reshape((-1, self.second_mult, self.gpt_model_dim))
        
        return AudioLDM2ProjectionModelOutput(
            hidden_states=hidden_states,
        )


class AudioLDM2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.ClapModel`]):
            First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model
            [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection),
            specifically the [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant. The
            text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to
            rank generated waveforms against the text prompt by computing similarity scores.
        text_encoder_2 ([`~transformers.T5EncoderModel`]):
            Second frozen text-encoder. AudioLDM2 uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) variant.
        projection_model ([`AudioLDM2ProjectionModel`]):
            A trained model used to linearly project the hidden-states from the first and second text encoder models
            and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are
            concatenated to give the input to the language model.
        language_model ([`~transformers.GPT2Model`]):
            An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected
            outputs from the two text encoders.
        tokenizer ([`~transformers.RobertaTokenizer`]):
            Tokenizer to tokenize text for the first frozen text-encoder.
        tokenizer_2 ([`~transformers.T5Tokenizer`]):
            Tokenizer to tokenize text for the second frozen text-encoder.
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            Vocoder of class `SpeechT5HifiGan` to convert the mel-spectrogram latents to the final audio waveform.
        img_project_model (Optional[`ImageProjectionLayer`]):
            A model layer used to project image embeddings into the high-dimensional space. 
            Required to integrate visual context into the audio generation process.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapModel,
        text_encoder_2: T5EncoderModel,
        projection_model: MyAudioLDM2ProjectionModel,
        language_model: GPT2Model,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast],
        feature_extractor: ClapFeatureExtractor,
        unet: MyAudioLDM2UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
        img_project_model: Optional[ImageProjectionLayer] = None,
    ):
        super().__init__()        
        
        if img_project_model is None:
            img_project_model = ImageProjectionLayer()
            print("Image projection layer added to the pipe!")
            if os.path.exists(IMG_PROJ_LAYER_WEIGHTS):
                # Load weights if found
                layer_weights = torch.load(IMG_PROJ_LAYER_WEIGHTS)
                img_project_model.load_state_dict(layer_weights)
                print("Image projection layer weights have been loaded!")
            else:
                print("No weights found for the image projection layer!")
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            img_project_model=img_project_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = [
            self.text_encoder.text_model,
            self.text_encoder.text_projection,
            self.text_encoder_2,
            self.projection_model,
            self.img_project_model,
            self.language_model,
            self.unet,
            self.vae,
            self.vocoder,
            self.text_encoder,
        ]

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """
        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)
            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def encode_prompt(
        self,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        prompt=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
        Returns:
            prompt_embeds (`torch.FloatTensor`):
                Text embeddings from the Flan T5 model.
            attention_mask (`torch.LongTensor`):
                Attention mask to be applied to the `prompt_embeds`.
            generated_prompt_embeds (`torch.FloatTensor`):
                Text embeddings generated from the GPT2 langauge model.

        Example:

        ```python
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get text embedding vectors
        >>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass text embeddings to pipeline for text-conditional audio generation
        >>> audio = pipe(
        ...     prompt_embeds=prompt_embeds,
        ...     attention_mask=attention_mask,
        ...     generated_prompt_embeds=generated_prompt_embeds,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ... ).audios[0]

        >>> # save generated audio sample
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```"""
        
        # Check if both the text and the text embedding are None
        # - If both conditions are True, it means that an image is given in input
        # - If one of the two conditions is False, it means that either text or text embedding is given in input
        

        # Assess if both image and text are provided in input as conditions for generation        
        if prompt is not None and image_embeds is not None:
            use_img = True
            use_txt = True
        elif prompt_embeds is not None and image_embeds is not None:
            use_img = True
            use_txt = True
        else:
            # Check if text/image is provided in input
            if not (prompt is None and prompt_embeds is None):
                use_img = False
                use_txt = True
            else:
                use_img = True
                use_txt = False
            
        if use_img and use_txt:
            # Text provided in input
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            # Text embedding provided in input
            else:
                batch_size = prompt_embeds.shape[0]
                
            # Image embedding in input
            if image_embeds is not None:
                img_batch_size = image_embeds.shape[0]
        else:
            if use_txt:
                # Text provided in input
                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)
                # Text embedding provided in input
                else:
                    batch_size = prompt_embeds.shape[0]
            else:
                # Image embedding in input
                if image_embeds is not None:
                    img_batch_size = image_embeds.shape[0]
                    batch_size = image_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        
        prompt_attention_mask = None
        proj_prompt_embeds = None
        proj_prompt_att_mask = None 
        
        img_attention_mask = None
        proj_image_embeds = None
        proj_img_att_mask = None
        
        if use_txt:
            if prompt_embeds is None:
                prompt_embeds_list = []
                attention_mask_list = []

                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length" if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True,
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids
                    attention_mask = text_inputs.attention_mask
                    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
                    

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                    ):
                        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                        logger.warning(
                            f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                            f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    text_input_ids = text_input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    

                    if text_encoder.config.model_type == "clap":
                        prompt_embeds = text_encoder.get_text_features(
                            text_input_ids,
                            attention_mask=attention_mask,
                        )
                        # prompt_embeds shape here: [1,512] || attention_mask here: [1,512]
                        # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                        prompt_embeds = prompt_embeds[:, None, :]
                        # make sure that we attend to this single hidden-state
                        attention_mask = attention_mask.new_ones((batch_size, 1))
                        # prompt_embeds shape here: [1,1,512] || attention_mask here: [1,1]
                    else:
                        prompt_embeds = text_encoder(
                            text_input_ids,
                            attention_mask=attention_mask,
                        )

                        prompt_embeds = prompt_embeds[0]
                        # prompt_embeds shape here: [1,15,1024] || attention_mask here: [1,15]

                    prompt_embeds_list.append(prompt_embeds)
                    attention_mask_list.append(attention_mask)

                projection_output = self.projection_model(
                    hidden_states=prompt_embeds_list[0],
                    hidden_states_1=prompt_embeds_list[1],
                    attention_mask=attention_mask_list[0],
                    attention_mask_1=attention_mask_list[1],
                )
                proj_prompt_embeds = projection_output.hidden_states
                proj_prompt_att_mask = projection_output.attention_mask
                
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
                prompt_attention_mask = (
                    attention_mask.to(device=device)
                    if attention_mask is not None
                    else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
                )
                                
        if use_img:
            image_embeds = image_embeds[:, None, :]
            img_attention_mask = torch.ones((img_batch_size, 1))
            proj_image_embeds, proj_img_att_mask = self.__encode_artwork_no_t5__(image_embeds, img_batch_size, device)
            
            image_embeds = image_embeds.to(dtype=EMBEDS_DTYPE, device=device)
            img_attention_mask = img_attention_mask.to(dtype=ATT_MASK_DTYPE, device=device)
            
            
        if proj_prompt_embeds is not None and proj_image_embeds is not None:
            projected_embeds = torch.cat([proj_prompt_embeds, proj_image_embeds], dim=1)
            projected_attention_mask = torch.cat([proj_prompt_att_mask, proj_img_att_mask], dim=1)
        elif proj_prompt_embeds is not None and proj_image_embeds is None:
            projected_embeds = proj_prompt_embeds
            projected_attention_mask = proj_prompt_att_mask
        else:
            projected_embeds = proj_image_embeds
            projected_attention_mask = proj_img_att_mask
        
        generated_prompt_embeds = self.generate_language_model(
                projected_embeds,
                attention_mask=projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )
        
        # Concat embeddings >>> Need this at inference time
        if prompt_embeds is not None and image_embeds is not None:
            input_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, img_attention_mask], dim=1)
        elif prompt_embeds is not None and image_embeds is None:
            input_embeds = prompt_embeds
            attention_mask = prompt_attention_mask
        else:
            input_embeds = image_embeds
            attention_mask = img_attention_mask    
            
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.language_model.dtype, device=device)

        bs_embed, seq_len, hidden_size = input_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        input_embeds = input_embeds.repeat(1, num_waveforms_per_prompt, 1)
        input_embeds = input_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, hidden_size)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_waveforms_per_prompt, seq_len)
        
        bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
        # duplicate generated embeddings for each generation per prompt, using mps friendly method
        generated_prompt_embeds = generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        generated_prompt_embeds = generated_prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )
        
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            # Added condition: check if prompt is not none, otherwise go on
            # Error raised when i provide an image in input w/o this condition
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif use_txt and batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            negative_attention_mask_list = []
            max_length = input_embeds.shape[1] # prev: max_length = prompt_embeds.shape[1]
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=tokenizer.model_max_length
                    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
                    else max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                uncond_input_ids = uncond_input.input_ids.to(device)
                negative_attention_mask = uncond_input.attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    negative_prompt_embeds = text_encoder.get_text_features(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    negative_attention_mask = negative_attention_mask.new_ones((batch_size, 1))
                else:
                    negative_prompt_embeds = text_encoder(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_attention_mask_list.append(negative_attention_mask)

            projection_output = self.projection_model(
                hidden_states=negative_prompt_embeds_list[0],
                hidden_states_1=negative_prompt_embeds_list[1],
                attention_mask=negative_attention_mask_list[0],
                attention_mask_1=negative_attention_mask_list[1],
            )
            negative_projected_prompt_embeds = projection_output.hidden_states
            negative_projected_attention_mask = projection_output.attention_mask

            negative_generated_prompt_embeds = self.generate_language_model(
                negative_projected_prompt_embeds,
                attention_mask=negative_projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
                dtype=self.language_model.dtype, device=device
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            negative_attention_mask = negative_attention_mask.repeat(1, num_waveforms_per_prompt)
            negative_attention_mask = negative_attention_mask.view(batch_size * num_waveforms_per_prompt, seq_len)

            # duplicate unconditional generated embeddings for each generation per prompt
            seq_len = negative_generated_prompt_embeds.shape[1]
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.view(
                batch_size * num_waveforms_per_prompt, seq_len, -1
            )
 
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            """Replaced prompt_embeds with input_embeds so that it works with either image or text embeddings."""
            input_embeds = torch.cat([negative_prompt_embeds, input_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask])
            generated_prompt_embeds = torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds])
            
        """Replaced prompt embeds with input_embeds"""
        
        if input_embeds.dtype != EMBEDS_DTYPE:
            input_embeds = input_embeds.to(EMBEDS_DTYPE)
            
        if attention_mask.dtype != ATT_MASK_DTYPE:
            attention_mask = attention_mask.to(ATT_MASK_DTYPE)
                
        return input_embeds, attention_mask, generated_prompt_embeds

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    """ ##################### MODIFIED THIS METHOD ##################### """
    def score_waveforms(self, text, image_emb, audio, num_waveforms_per_prompt, device, dtype):
        if not is_librosa_available():
            logger.info(
                "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
            )
            return audio
        
        if text is not None and image_emb is None:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            resampled_audio = librosa.resample(
                audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
            )
            inputs["input_features"] = self.feature_extractor(
                list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
            ).input_features.type(dtype)
            inputs = inputs.to(device)

            # compute the audio-text similarity score using the CLAP model
            logits_per_text = self.text_encoder(**inputs).logits_per_text
            # sort by the highest matching generations per prompt
            indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
            audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
            
        elif text is None and image_emb is not None:
            # Compute best audio with respect to Image by computing ImageBind embedding similarity.
            tmp = audio.numpy()
            
            # Load ImageBind
            imagebind = load_model(False)
            
            # Store files so to compute embeddings
            for i in range(len(tmp)):
                scipy.io.wavfile.write(TEMP_AUDIO_DIR + f"tmp_aud_{i}.wav", rate=16000, data=tmp[i])
            audio_file = [TEMP_AUDIO_DIR + file for file in os.listdir(TEMP_AUDIO_DIR)]
            
            # Generate audio embeddings using ImageBind
            audio_emb = generate_embeds(imagebind, audio_paths=audio_file, extract_emb=True, emb_type='audio')
            audio_emb = [t.numpy() for t in audio_emb]
            
            # Remove files after computing embeddings
            for file in audio_file:
                os.remove(file)
            
            # Compute similiarties among embeddings
            tmp_imb_emb = image_emb.reshape(1, -1)
            similarities = cosine_similarity(tmp_imb_emb, audio_emb)
        
            # Sort audio_emb indices based on similarity values in descending order
            sorted_indices = np.argsort(-similarities, axis=1)
            
            # Sort audio based on sorted_indices
            audio = [audio[i] for i in sorted_indices][0]
                  
        elif text is not None and image_emb is not None:
            # Compute best audio with respect to Image and Text by computing ImageBind embedding similarity.
            tmp = audio.numpy()
            
            # Load ImageBind
            imagebind = load_model(False)
            
            # Store files so to compute embeddings
            for i in range(len(tmp)):
                scipy.io.wavfile.write(TEMP_AUDIO_DIR + f"tmp_aud_{i}.wav", rate=16000, data=tmp[i])
            audio_file = [TEMP_AUDIO_DIR + file for file in os.listdir(TEMP_AUDIO_DIR)]
            
            # Generate audio and text embeddings using ImageBind
            audio_emb = generate_embeds(imagebind, audio_paths=audio_file, extract_emb=True, emb_type='audio')
            text_emb = generate_embeds(imagebind, text=text, extract_emb=True, emb_type='text')
            audio_emb = [t.numpy() for t in audio_emb]
            text_emb = text_emb.numpy()
            
            # Sum up image and text embeddings
            img_txt_emb = image_emb + text_emb
            
            # Remove files after computing embeddings
            for file in audio_file:
                os.remove(file)
            
            # Compute similiarties among embeddings
            img_txt_emb = img_txt_emb.reshape(1, -1)
            similarities = cosine_similarity(img_txt_emb, audio_emb)
        
            # Sort audio_emb indices based on similarity values in descending order
            sorted_indices = np.argsort(-similarities, axis=1)
            
            # Sort audio based on sorted_indices
            audio = [audio[i] for i in sorted_indices][0]

        else:
            return audio
                   
        return audio

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    """ ##################### MODIFIED THIS METHOD ##################### """
    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        image_embeds=None,
        negative_prompt_embeds=None,
        generated_prompt_embeds=None,
        negative_generated_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,
    ):
        
        # Check if both the text and the text embedding are None
        # - If both conditions are True, it means that an image is given in input
        # - If one of the two conditions is False, it means that either text or text embedding is given in input
        is_text_input = not (prompt is None and prompt_embeds is None)
        
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if is_text_input:
            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt is None and (prompt_embeds is None or generated_prompt_embeds is None):
                raise ValueError(
                    "Provide either `prompt`, or `prompt_embeds` and `generated_prompt_embeds`. Cannot leave "
                    "`prompt` undefined without specifying both `prompt_embeds` and `generated_prompt_embeds`."
                )
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
                if attention_mask is not None and attention_mask.shape != prompt_embeds.shape[:2]:
                    raise ValueError(
                        "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                        f"`attention_mask: {attention_mask.shape} != `prompt_embeds` {prompt_embeds.shape}")
        
        else:
            if image_embeds is not None:
                if attention_mask is not None and attention_mask.shape != image_embeds.shape[:2]:
                    raise ValueError(
                        "`attention_mask should have the same batch size and sequence length as `image_embeds`, but got:"
                        f"`attention_mask: {attention_mask.shape} != `image_embeds` {image_embeds.shape}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_embeds is not None and negative_generated_prompt_embeds is None:
            raise ValueError(
                "Cannot forward `negative_prompt_embeds` without `negative_generated_prompt_embeds`. Ensure that"
                "both arguments are specified"
            )

        if generated_prompt_embeds is not None and negative_generated_prompt_embeds is not None:
            if generated_prompt_embeds.shape != negative_generated_prompt_embeds.shape:
                raise ValueError(
                    "`generated_prompt_embeds` and `negative_generated_prompt_embeds` must have the same shape when "
                    f"passed directly, but got: `generated_prompt_embeds` {generated_prompt_embeds.shape} != "
                    f"`negative_generated_prompt_embeds` {negative_generated_prompt_embeds.shape}."
                )
            if (
                negative_attention_mask is not None
                and negative_attention_mask.shape != negative_prompt_embeds.shape[:2]
            ):
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {negative_attention_mask.shape} != `prompt_embeds` {negative_prompt_embeds.shape}"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
    
        return latents
    
    """
    #############################
    ######### INFERENCE #########
    #############################
    """
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        image_path: Optional[str] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
        artwork_text_datasets: Optional[Union[str, List[str]]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
            audio_length_in_s (`int`, *optional*, defaults to 10.24):
                The length of the generated audio sample in seconds.
            num_inference_steps (`int`, *optional*, defaults to 200):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                A higher guidance scale value encourages the model to generate audio that is closely linked to the text
                `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt. If `num_waveforms_per_prompt > 1`, then automatic
                scoring is performed between the generated outputs and the text prompt. This scoring ranks the
                generated waveforms based on their cosine similarity with the text input in the joint text-audio
                embedding space.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            image_path (`str`, *optional*):
                The path of the image to use to generate music. If not provided, either image_embeds, prompt 
                or prompt_embeds must be provided.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated image embeddings. Can be used to condition the generation process. If not
                provided, either image_path, prompt or prompt_embeds must be provided.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will
                be taken from the config of the model.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or
                `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion
                model (LDM) output.
            artwork_text_datasets (`str` or `List[str]`, *optional*):
                The dataset to which the artwork's audio belongs to.
                If an image is given in input, this must be given along with it.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated audio.
        """
        # Determine if text and text embedding have not been provided in input None
        # True if either text or text embedding was given in input, False otherwise
        device = self._execution_device
        is_text_input = not (prompt is None and prompt_embeds is None)
        use_music_prompt = not (image_embeds is None or image_path is None)
        
        if image_path is not None and image_embeds is None:
            imagebind = load_model(full_log=False, use_cpu=True)
            image_embeds = generate_embeds(imagebind, 'cpu', [image_path], extract_emb=True, emb_type='vision')[0]
            image_embeds = image_embeds.unsqueeze(0).to(device)
        
        if self.img_project_model.device.type == 'cpu' and device.type == 'cuda':
            self.img_project_model = self.img_project_model.to(device)
        
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Define call parameters
        if is_text_input:
            # Text in input
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
        else:
            # Image in input
            batch_size = image_embeds.shape[0]
            
        # If image_embeds have been provided in input, prepare prompt based on batch_size
        if not is_text_input:
            prompt = []
            # Inference with CombinedDataset
            if artwork_text_datasets is not None:
                for art_datas in artwork_text_datasets:
                    if art_datas == "BASIC":
                        prompt.append(BASIC_DATASET_TEXT)
                    elif art_datas == "HISTORICAL":
                        prompt.append(HISTORICAL_DATASET_TEXT)
                    else:
                        prompt.append(EMOTIONAL_DATASET_TEXT)
            else:
                # Inference with ImageAudioDataset
                prompt = [BASIC_DATASET_TEXT] * batch_size  
            
            if len(prompt) == 1:
                prompt = prompt[0]

        if use_music_prompt:
            music_prompt = []
            
            if artwork_text_datasets is not None:
                for art_datas in artwork_text_datasets:
                    if art_datas == "BASIC":
                        music_prompt.append(BASIC_DATASET_TEXT)
                    elif art_datas == "HISTORICAL":
                        music_prompt.append(HISTORICAL_DATASET_TEXT)
                    else:
                        music_prompt.append(EMOTIONAL_DATASET_TEXT)
            else:
                music_prompt = [BASIC_DATASET_TEXT] * batch_size
                
            if len(music_prompt) == 1:
                music_prompt = music_prompt[0]

        # 2. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            image_embeds,
            negative_prompt_embeds,
            generated_prompt_embeds,
            negative_generated_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )
        
        # 3. Assess if CPU is being used
        if str(device) == 'cpu':
            global EMBEDS_DTYPE
            EMBEDS_DTYPE = torch.float32
        
        # 4. Determine if classifier free guidance should be used
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
                            
        # 5. Encode input prompt
        input_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            image_embeds=image_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generated_prompt_embeds=generated_prompt_embeds,
            negative_generated_prompt_embeds=negative_generated_prompt_embeds,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            max_new_tokens=max_new_tokens,
        )
        
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            input_embeds.dtype,
            device,
            generator,
            latents,
        )
                
        # 8. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        if use_music_prompt:
            # Encode music prompt using T5
            input_embeds , attention_mask = self.__encode_music_prompt_t5__(music_prompt, negative_prompt, do_classifier_free_guidance,
                                                                            num_waveforms_per_prompt, batch_size, device)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if device == 'cpu':
                    latent_model_input.to(dtype=EMBEDS_DTYPE)
                    
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=input_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        self.maybe_free_model_hooks()

        # 10. Post-processing
        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            mel_spectrogram = self.vae.decode(latents).sample
        else:
            return AudioPipelineOutput(audios=latents)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

        audio = audio[:, :original_waveform_length]

        
        # 11. Automatic scoring
        if num_waveforms_per_prompt > 1 and prompt is not None:
            audio = self.score_waveforms(
                text=prompt,
                image_emb=None,
                audio=audio,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
                dtype=input_embeds.dtype,
            )
        elif num_waveforms_per_prompt > 1 and image_embeds is not None:
            audio = self.score_waveforms(
                text=None,
                image_emb=image_embeds,
                audio=audio,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
                dtype=input_embeds.dtype,
            )

        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)

    """
    ############################
    ######### TRAINING #########
    ############################
    """
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __train__(
        self,
        prompt: Union[str, List[str]] = None,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        timesteps: Optional[torch.LongTensor] = None,
        artwork_text_datasets: Optional[Union[str, List[str]]] = None,
    ):
        r"""
        Training procedure of the pipeline for audio generation based on the given parameters.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                A higher guidance scale value encourages the model to generate audio that is closely linked to the text
                `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt. If `num_waveforms_per_prompt > 1`, then automatic
                scoring is performed between the generated outputs and the text prompt. This scoring ranks the
                generated waveforms based on their cosine similarity with the text input in the joint text-audio
                embedding space.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated image embeddings. Can be used to condition the generation process. If not
                provided, either prompt or prompt_embeds must be provided.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will
                be taken from the config of the model.
            timesteps (`torch.LongTensor`, *optional*):
                Timesteps for denoising process.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            artwork_text_datasets (`str` or `List[str]`, *optional*):
                The dataset to which the artwork's audio belongs to.
                If an image is given in input, this must be given along with it.
                
        Examples:
        
        Returns:
            - noise_pred:
                A `torch.FloatTensor` consisting of the predicted noise residual generated by the UNet.
            - generated_prompt_embeds:
                A `torch.FloatTensor` consisting of GPT2's generated prompt embeddings.
        """
        # Determine if text and text embedding have not been provided in input None
        # True if either text or text embedding was given in input, False otherwise
        is_text_input = not (prompt is None and prompt_embeds is None)      
        use_music_prompt = not (image_embeds is None)

        # 1. Define batch size
        if is_text_input:
            # Text in input
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
        else:
            # Image in input
            batch_size = image_embeds.shape[0]
            
        # 2. Assess if CPU is being used
        device = self._execution_device
        if str(device) == 'cpu':
            global EMBEDS_DTYPE
            EMBEDS_DTYPE = torch.float32
            
        if self.img_project_model.device.type == "cpu" and device != 'cpu':
            self.img_project_model = self.img_project_model.to(device=device)
        
        # 3. Determine if classifier free guidance should be used
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 4. Encode text resembling our goal (generate music from artworks)
        #    If image_embeds have been provided in input, prepare prompt based on batch_size
        if not is_text_input:
            prompt = []
            # Training with CombinedDataset
            if artwork_text_datasets is not None:
                for art_datas in artwork_text_datasets:
                    if art_datas == "BASIC":
                        prompt.append(BASIC_DATASET_TEXT)
                    elif art_datas == "HISTORICAL":
                        prompt.append(HISTORICAL_DATASET_TEXT)
                    else:
                        prompt.append(EMOTIONAL_DATASET_TEXT)
            else:
                # Training with ImageAudioDataset
                prompt = [BASIC_DATASET_TEXT] * batch_size  
                
        if use_music_prompt:
            music_prompt = []
            
            if artwork_text_datasets is not None:
                for art_datas in artwork_text_datasets:
                    if art_datas == "BASIC":
                        music_prompt.append(BASIC_DATASET_TEXT)
                    elif art_datas == "HISTORICAL":
                        music_prompt.append(HISTORICAL_DATASET_TEXT)
                    else:
                        music_prompt.append(EMOTIONAL_DATASET_TEXT)
            else:
                music_prompt = [BASIC_DATASET_TEXT] * batch_size
            
        if generated_prompt_embeds is None:
            input_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
                device,
                num_waveforms_per_prompt,
                do_classifier_free_guidance,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                image_embeds=image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                generated_prompt_embeds=generated_prompt_embeds,
                negative_generated_prompt_embeds=negative_generated_prompt_embeds,
                attention_mask=attention_mask,
                negative_attention_mask=negative_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        # 5. If the model is on GPU and the latents are not, move them to GPU
        if latents.device.type == "cpu" and device != 'cpu':
            latents = latents.to(device=device)
            
        # 6. Prepare UNet's input
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps)

        if device == 'cpu':
            latent_model_input.to(dtype=EMBEDS_DTYPE)

        if use_music_prompt:
            # Encode music prompt using T5
            input_embeds , attention_mask = self.__encode_music_prompt_t5__(music_prompt, negative_prompt, do_classifier_free_guidance,
                                                                            num_waveforms_per_prompt, batch_size, device)

        timesteps = torch.cat([timesteps, timesteps]) if do_classifier_free_guidance else timesteps

        # 7. Predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=input_embeds,
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
        )[0]
        
        # 8. If needed, perform classifier free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        return noise_pred, generated_prompt_embeds
    
   
    def __encode_artwork_no_t5__(self, artwork_embeds, batch_size, device):
        """
        Applies linear transformation on an artwork embedding, and optionally applies class-free guidance.

        Params:
            - artwork_embeds (torch.Tensor): The embeddings of the artwork to be encoded.
            - batch_size (int): The size of the batch being processed.
            - device (torch.device): The device (CPU or GPU) on which to perform the computations.

        Returns:
            - tuple: A tuple containing:
                - input_embeds (torch.Tensor): The encoded artwork embeddings, ready for further processing.
                - attention_mask (torch.Tensor): The attention mask corresponding to the encoded embeddings.
        """
        # Linear projection of the image's embedding
        projection_output = self.img_project_model(
            hidden_states=artwork_embeds,
        )
        proj_hidden_states = projection_output.hidden_states
        proj_img_att_mask = torch.ones((batch_size, 1))
        
        input_embeds = proj_hidden_states.to(dtype=EMBEDS_DTYPE, device=device)
        attention_mask = proj_img_att_mask.to(dtype=ATT_MASK_DTYPE, device=device)
        
        return input_embeds, attention_mask
      
    
    def __encode_music_prompt_t5__(self, prompt, neg_prompt, class_guidance, no_wavs, b_size, device):
        """
        Encodes music prompts using a T5 encoder, optionally including negative prompts for class guidance.

        Params:
            - prompt (str or List[str]): The music prompt(s) to be encoded.
            - neg_prompt (str or List[str], optional): The negative prompt(s) for class guidance.
            - class_guidance (bool): If True, includes negative prompts for class guidance.
            - no_wavs (int): The number of waveforms to be processed.
            - b_size (int): The batch size for the prompts.
            - device (torch.device): The device (CPU or GPU) on which to perform the computations.

        Returns:
            - tuple: A tuple containing:
                - input_embeds (torch.Tensor): The encoded music prompt embeddings.
                - attention_mask (torch.Tensor): The attention mask corresponding to the encoded embeddings.
        """
        t5_tokenizer = self.tokenizer_2
        t5_encoder = self.text_encoder_2
        
        # Prompt embeddng
        text_inputs = t5_tokenizer(
            prompt,
            padding=True,
            max_length=t5_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        text_input_ids = text_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prompt_embeds = t5_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        
        p_embeds = prompt_embeds[0]
        att_mask = attention_mask
        
        p_embeds = p_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        att_mask = (
            att_mask.to(device=device)
            if att_mask is not None
            else torch.ones(p_embeds.shape[:2], dtype=torch.long, device=device)
        )

        bs_embed, seq_len, hidden_size = p_embeds.shape
        p_embeds = p_embeds.repeat(1, no_wavs, 1)
        p_embeds = p_embeds.view(bs_embed * no_wavs, seq_len, hidden_size)

        p_attention_mask = att_mask.repeat(1, no_wavs)
        p_attention_mask = p_attention_mask.view(bs_embed * no_wavs, seq_len)

        # Negative prompt embedding
        if class_guidance:
            uncond_tokens: List[str]
            if neg_prompt is None:
                uncond_tokens = [""] * b_size
            elif prompt is not None and type(prompt) is not type(neg_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(neg_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(neg_prompt, str):
                uncond_tokens = [neg_prompt]
            else:
                uncond_tokens = neg_prompt

            max_length = p_embeds.shape[1]
                
            uncond_input = t5_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = t5_encoder(
                uncond_input_ids,
                attention_mask=negative_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, no_wavs, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(b_size * no_wavs, seq_len, -1)

            negative_attention_mask = negative_attention_mask.repeat(1, no_wavs)
            negative_attention_mask = negative_attention_mask.view(b_size * no_wavs, seq_len)

            input_embeds = torch.cat([negative_prompt_embeds, p_embeds])
            attention_mask = torch.cat([negative_attention_mask, p_attention_mask])
        
        return input_embeds, attention_mask