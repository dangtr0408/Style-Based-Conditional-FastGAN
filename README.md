Developing this project has greatly expanded my knowledge in both academic research and coding for state-of-the-art GAN models. However, it has become quite time-consuming, so I’ve decided to move on to a new project. I’m leaving the code here in hopes that it may be helpful for others on their research journey. :)

Note:
- The goal of this project is to create a generator that is fast to train yet efficient to invert (see GAN inversion papers for more details).
- I removed the conditional option later in the project to focus on improving the model's image quality.
- This implementation achieves an FID of 42.5 for art paintings (1k images) at a resolution of 256x256.

Highlight features:
- Style-based learning using a mapping network and AdaIN, inspired by StyleGAN1.
- Wavelet extraction to enhance high-frequency features.
- Support for Projected GAN and PyTorch DDP.


Before training make sure your image folders are in the following format:

```
images_folder
├── cat_folder  (class1)
├── dog_folder  (class2)
├── bird_folder (class3)
└── ...         (class n)
```

Then your images directory will be: "D:\YourDir\images_folder"

How to train:

```batch
python train.py --dir D:\YourDir\YourFolder --n_gpus 1 --im_size 512 --batch_size 8 --inf
```
