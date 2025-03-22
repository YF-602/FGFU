dataset = "clintox"
dataset_seed = 1
cuda_device_num = 0
num_tasks = None

# here is the num_tasks para
if dataset in ["bace", "bbbp", "hiv"]:
    num_tasks = 1
elif dataset == "clintox":
    num_tasks = 2
elif dataset == "tox21":
    num_tasks = 12
elif dataset == "toxcast":
    num_tasks = 617
elif dataset == "sider":
    num_tasks = 27
elif dataset == "muv":
    num_tasks = 17
else:
    num_tasks = -1


fgfu_config = {
    "conv_config": {
        "hidden_dim": 64, # 2.must == emb_dim
        "mlp1_layers": 2,
        "mlp2_layers": 2,
        "mlp3_layers": 2,
        "mlp4_layers": 2,
        "aggr": "mean",
        "dropout": 0,
        "normalization": "bn",
        "input_norm": False,
    },
    "mlp_config": {
        "num_layers": 3,
        "in_channels": 128, # 3.must 2 * emb_dim
        "hidden_channels": 64,
        "out_channels": num_tasks,
        "dropout": 0,
        "normalization": "bn",
    },
    "emb_dim": 64, # 1.emb_dim
    "All_num_layers": 5,
    "activation": "relu",
    "dropout": 0,
}


trainer_fgfu_config = {
    "seed": dataset_seed,
    "cuda_device": cuda_device_num,
    "batch_size": 64, # 小数据集则64
    "lr": 4e-5,
    "epoch": 300,
    "dataset": dataset,
    "weight_decay": 1e-3,
    "model_type": "fgfu",
    "early_stop_patience": 30,
    "comments": "train fgfu",
    "save_id": 1,
}
