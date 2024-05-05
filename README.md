
# Food-Images-Generator

- a food image generator using WGAN-GP trained on food-101 data set which contains a 101 food categories each have 1000 food images.

- the model was trained for 6 hours (600 epochs) using RTX 4060.

- the model was also tested on Fashion and MNIST datasets and has given a good result for both.

- the model is built according to the [WGAN_GP paper](https://arxiv.org/pdf/1704.00028v3) with almost the same parameters but with changes in the Discriminator and Generator architectures.

- the results on Food-101 dataset :

original image:

![screenshot](results/original_image.png)

generated image:

![screenshot](results/generated_image.png)

   Generator Loss:

![screenshot](results/Generator_Loss.png)

   Discriminator Loss:

![screenshot](results/Discriminator_Loss.png)