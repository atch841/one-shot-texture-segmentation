# Most of the code here are taken from https://github.com/ivust/one-shot-texture-segmentation

import numpy as np

def generate_collages(
        textures,
        batch_size=1,
        segmentation_regions=10,
        anchor_points=None):
    # Returns a batch of mixed texture, reference mask, and reference texture index
    N_textures = textures.shape[0]
    img_size= textures.shape[1]
    masks, n_points = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    batch = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    ref_idx = [np.random.randint(i) for i in n_points]
    return batch, masks[range(batch_size),:,:,ref_idx].reshape((batch_size, 256, 256, 1)), textures_idx[ref_idx,range(batch_size)]

def generate_random_masks(img_size=256, batch_size=1, segmentation_regions=10, points=None):
    xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size))

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
        # n_points = [segmentation_regions] * batch_size
        points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
        
    masks = []
    for b in range(batch_size):
        dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]]
        voronoi = np.argmin(dists_b, axis=0)
        masks_b = np.zeros((img_size, img_size, segmentation_regions))
        for m in range(segmentation_regions):
            masks_b[:,:,m][voronoi == m] = 1
        masks.append(masks_b)
    return np.stack(masks), n_points

def generate_validation_collages(N=50):
    textures = np.load('validation_textures.npy')
    collages = generate_collages(textures, batch_size=N)
    np.save('validation_collages.npy', collages)

if __name__ == '__main__':
    generate_validation_collages()
