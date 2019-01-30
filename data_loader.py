from generate_collages import generate_collages
import numpy as np

class Data_loader:
	def __init__(self, textures_path, batch_size, max_region=10):
		textures = np.load(textures_path)
		self.textures = textures
		self.batch_size = batch_size
		self.max_region = max_region
		self.ref_texture = textures[:,96:160,96:160,:]
	def get_batch_data(self):
		# generate mixed texture image, reference patch, and reference mask
		batch, mask, ref_ind = generate_collages(self.textures, self.batch_size, self.max_region)
		ref_patch = self.ref_texture[ref_ind]
		return batch/255, mask, ref_patch/255