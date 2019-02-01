# one-shot-texture-segmentation
This is a re-implementation of one shot texture segmentation https://arxiv.org/abs/1807.02654

Part of the code are taken from the original author from https://github.com/ivust/one-shot-texture-segmentation

DTD dataset can be downloaded from https://www.robots.ox.ac.uk/~vgg/data/dtd/

# Training step
1. Run `generate_texture.py` to generate train_texture.npy and val_texture.npy
2. Run `train.py`

# Note
This re-implementation is currently not finished. The model is still underfitting. It might because of the vgg layers should be initialized with pretrained weight. Not planning to update this repo anymore, but anyone interested can carry on this work.