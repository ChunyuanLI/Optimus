import torch
import os

output_dir = "../output/philly_rr1_vae_wikipedia_pretraining_2nd_file"

data = torch.load(os.path.join(output_dir, 'batch_debug_6621.pt')