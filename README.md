Ongoing project...

What's new:
- Added style-based learning using mapping network and adain similar to StyleGAN1.
- Added conditional by modify the SLE layers.
- Switch to pytorch DDP.

Before training make sure your image folders are as following format:
images_folder
├── cat_images (class1)
├── dog_images (class2)
└── cat_images (class1)
Then your images dir will be: D:\YourDir\images_folder

How to train:

```batch
python train.py --dir --inf
```
You can enable conditional training anytime by adding "--cond".
