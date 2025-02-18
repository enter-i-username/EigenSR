
# common
exp_folder = './experiments/'
device = 'cuda'
device_ids = [0, 1]

# model
vit_body_fn = './pretrained/IPT_sr2.pt'
scale = 2
hidden_channels = 64
need_lora = True
lora_r = 4

# dataset
dataset_root = './your_own_datasets'

train_dataset = 'ARAD1K'  # 'NWPU RESISC45' 'ARAD1K'
train_crop_size = (scale * 48, scale * 48)
train_random_flip_p = 0.5
train_svd_threshold = 0.97  # see the appendix of our paper on arxiv: https://arxiv.org/abs/2409.04050
train_batch_size = 64
train_workers = 0

test_dataset = 'ARAD1K'  # optional: None
test_batch_size = 8
test_workers = 0

dataset_cache = True

# training and val settings
num_epochs = 2000
lr = 0.001
lr_linear_decay = False

inference_batch_size = 64
inference_step_size = (48, 48)
inference_eigen_dims = 0.5  # 0 < R < 1: Ratio to the input channels. (float)| R >= 1: No. channels (int)| None: use all channels
metric_keys = ['mPSNR', 'mSAM']
metric_optimal_key = 'mPSNR'
metric_optimal = 'max'
test_every = 100

# save
loss_log_save_every = 1
loss_log_fn = 'loss_log/loss_log_{epoch}.json'
metric_log_save_every = 100
metric_log_fn = 'metric_log/metric_log_{epoch}.json'
optimizer_save_every = 100
optimizer_fn = 'optim/optim_{epoch}.pt'
trainable_save_every = 100
trainable_params_fn = 'trainable/trainable_{epoch}.pt'
best_trainable_fn = 'trainable/best.pt'
