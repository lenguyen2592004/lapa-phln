import torch

from laq_model import LAQTrainer
from laq_model import LatentActionQuantization


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).to(device)


trainer = LAQTrainer(
    laq,
    folder = '',
    offsets = 30,
    batch_size = 64,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 100005,
    results_folder='results',
    lr=1e-4,
    save_model_every=5000,
    save_results_every=5000,
)

trainer.train()        

