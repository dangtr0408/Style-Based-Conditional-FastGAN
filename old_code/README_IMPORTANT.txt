This is an older implementation that uses self-supervised discriminator, similar to the original FastGAN. I'll highlight some modifications to the discriminator here:

- The discriminator is slightly deeper at the beginning. Refer to the original FastGAN code for comparison.

- Smaller output size + minibatch std, this change might compromise performance on certain datasets, as the authors note that their discriminator performs better on art datasets with a larger output size.

- I'm using WGAN loss with R1 regularization. I find it converges faster than hinge loss with spectral normalization. The training speed is roughly the same because the discriminator is small (~8M parameters), so applying R1 regularization every 16 iterations is relatively cheap to compute.

- The decoder is trained using SSIM loss. In the original implementation, they used perceptual loss, which is computationally heavier. While SSIM loss produces comparable reconstruction quality, I haven't thoroughly tested whether it performs as well as perceptual loss.

- The FastGAN authors added several enhancements to their discriminator that aren't documented in their paper, including SLE, small/large discriminators, and three decoders. I removed all of these to keep the model as lightweight as possible. However, these additions do seem to improve image quality for some datasets.

While this setup converges at a reasonable speed, the overall quality is significantly inferior compare to projected discriminator. I’m leaving this here for anyone interested in further developing it.