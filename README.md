# Fine-Tune-Stable-Diffusion
We successfully fine tuned the base Stable Diffusion model to generate Renaissance Style Portraits!

If you'd like to see our process, results, or additional information regarding this project, please navigate to the Wiki section of this repository also available [here](https://github.com/martingasparyan/Fine-Tune-Stable-Diffusion/wiki).

For those who would like to test out our model or just generate images using it, follow the steps below.

1) Create a base Stable diffusion Model
```python
new_model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
 ```
2) Load Weights from our h5 model which is hosted on Hugging Face [here](https://huggingface.co/morj/renaissance/blob/main/stable_diffusion_renaissance.h5):
```python
new_model.diffusion_model.load_weights('/path/to/file/renaissance_model.h5')
```
3) Create a variable to hold the values of the to-be-generated image such as prompt, batch size, iterations, and seed
```python 
img = new_model.text_to_image(
       prompt="A woman with an enigmatic smile against a dark background",
       batch_size=1,  # How many images to generate at once
       num_steps=25,  # Number of iterations (controls image quality)
       seed=123,  # Set this to always get the same image from the same prompt
    )
```
4) Display using the function:
```python
def plot_images(images):
    plt.figure(figsize=(5, 5))
    plt.imshow(images)
    plt.axis("off")
    
plot_images(img)
```
