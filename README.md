Ongoing project...

What's new:
- Added style-based learning using a mapping network and AdaIN, similar to StyleGAN1.
- Wavelets extraction for high freq features.
- Support Projected GAN and PyTorch DDP.

Before training make sure your image folders are in the following format:

```
images_folder
├── cat_images  (class1)
├── dog_images  (class2)
├── bird_images (class3)
└── ...         (class n)
```

Then your images directory will be: "D:\YourDir\images_folder"

How to train:

```batch
python train.py --dir D:\YourDir\YourFolder --n_gpus 1 --im_size 512 --batch_size 8 --inf
```
