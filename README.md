# Renaissance Stable Diffusion 
**This model uses the KerasCV implementation of stability.ai's text-to-image model. Unlike other open-source alternatives like Hugging Face's Diffusers, KerasCV offers advantages such as XLA compilation and mixed precision support, resulting in state-of-the-art generation speed.**

- Baseline Stable Diffusion Model = Generated using Stability AI's Stable Diffusion 2.1
- Finetuned = Trained on 256x256 px with a batch size of 4, for 577 epochs
- Refined = Trained on 512x512 px, with a batch size of 1, for 72 epochs
1. "A young man."



<center>

 <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/young_man_base.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/young_man_256.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/young_man_512.png" width="256" height="256"> |
| --- | --- | --- |
| Baseline Stable Diffusion Model | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finetuned | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Refined |

</center>




2. "A young woman with blue eyes against a dark background."



| <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_blue_eyes_base.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_with_blue_eyes_256.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_blue_eyes_512.png" width="256" height="256"> |
| --- | --- | --- |
| Baseline Stable Diffusion Model | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finetuned | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Refined |



3. "A woman against a dark background looking mysterious."



| <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_dark_background_base.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_mysterious_dark_background_256.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/woman_dark_background512.png" width="256" height="256"> |
| --- | --- | --- |
| Baseline Stable Diffusion Model | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finetuned | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Refined |


4. "A beautiful horse running through a field."
<div align="center">

| <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/base_dziuk.png" width="256" height="256"> | <img src="https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/blob/main/Images/horse_512.png" width="256" height="256"> |
| --- | --- |
| Baseline Stable Diffusion Model |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Refined |
</div>

- If you'd like to see more regarding our process, results, or additional information about this project, please navigate to the Wiki section of this repository also available [here](https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/wiki).

**Model Details** 
- Model is available at https://huggingface.co/morj/renaissance
- Dataset is available at https://huggingface.co/datasets/morj/renaissance_portraits
- Developed by: Martin Gasparyan and Tatev Kyosababyan
- Model type: Diffusion-based text-to-image generative model
- Model Description: This model can be used to generate and modify images based on text prompts. It is a Latent Diffusion Model that uses a fixed, pretrained text encoder (OpenCLIP-ViT/H) to generate high-quality - Reniassance portraits from textual prompts. This model uses the KerasCV implementation of stability.ai's text-to-image model, Stable Diffusion. 
- License: CreativeML Open RAIL++-M License
- Finetuned from model: stabilityai/stable-diffusion-2-1


## To Generate your own Examples:

1) Install Dependencies
   ```python
   !pip install keras-cv==0.6.0 -q
   !pip install -U tensorflow -q
   !pip install keras-core -q
   ```
2) Imports
   ```python
   from textwrap import wrap
   import os
   import keras_cv
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import tensorflow.experimental.numpy as tnp
   from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
   from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
   from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
   from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
   from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
   from tensorflow import keras
   ```
3) Create a base Stable diffusion Model
```python
my_base_model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
 ```
4) Load Weights from our h5 model which is hosted on Hugging Face [here](https://huggingface.co/morj/renaissance/blob/main/stable_diffusion_renaissance.h5):
```python
my_base_model.diffusion_model.load_weights('/path/to/file/renaissance_model.h5')
```
5) Create a variable to hold the values of the to-be-generated image such as prompt, batch size, iterations, and seed
```python 
img = my_base_model.text_to_image(
       prompt="A woman with an enigmatic smile against a dark background",
       batch_size=1,  # How many images to generate at once
       num_steps=25,  # Number of iterations (controls image quality)
       seed=123,  # Set this to always get the same image from the same prompt
    )
```
6) Display using the function:
```python
def plot_images(images):
    plt.figure(figsize=(5, 5))
    plt.imshow(images)
    plt.axis("off")
    
plot_images(img)
```


For more details, please check out the Wiki section of this repository also available [here](https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/wiki).
