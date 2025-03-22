from utils.hdata import RawMoleculeDataset
from trainer import Trainer
from models.fgfu import FGFU
import os
from config import (
    fgfu_config,
    trainer_fgfu_config,
)

cur_path = os.getcwd()
root_path = cur_path + "/dataset/{}/".format(trainer_fgfu_config["dataset"])

train_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_fgfu_config["seed"],
    mode="train",
    dataset=trainer_fgfu_config["dataset"],
)

valid_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_fgfu_config["seed"],
    mode="valid",
    dataset=trainer_fgfu_config["dataset"],
)

test_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_fgfu_config["seed"],
    mode="test",
    dataset=trainer_fgfu_config["dataset"],
)

print("DataSet Loaded! Current DataSet is {}".format(
    trainer_fgfu_config["dataset"]))


def train_fgfu_model():
    eval_dict = {}
    cur_fgfu_config = trainer_fgfu_config
    cur_fgfu_config["save_id"] = cur_fgfu_config["save_id"] * 100

    lr_list = [
        1e-2,
        8e-3,
        6e-3,
        4e-3,
        2e-3,
        1e-3,
        8e-4,
        6e-4,
        4e-4,
        2e-4,
        1e-4,
        8e-5,
        6e-5,
        4e-5,
        2e-5,
        1e-5,
    ]

    for cur_lr in lr_list:
        fgfu_model = FGFU(fgfu_config)
        cur_fgfu_config["lr"] = cur_lr
        print("cur lr is {:.6f}".format(cur_lr))
        trainer = Trainer(
            fgfu_model, cur_fgfu_config, train_dataset, valid_dataset, test_dataset
        )
        trainer.train()
        auc_mean, auc_std = trainer.eval_model()
        print(
            "cur lr: {} ; auc mean: {}; auc std: {}".format(
                cur_lr, auc_mean, auc_std)
        )
        eval_dict[cur_fgfu_config["save_id"]] = (cur_lr, auc_mean, auc_std)
        cur_fgfu_config["save_id"] = cur_fgfu_config["save_id"] + 1

    for key in eval_dict.keys():
        print(key, eval_dict[key])

if __name__ == "__main__":
    train_fgfu_model()
