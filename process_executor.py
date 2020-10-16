import os
import yaml
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import argparse

def pick_best(tb_path):

    print("tb_path:", tb_path)

    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()

    hmean_list = ea.scalars.Items("results/hmean")

    value_list = []
    step_list = []

    for item in hmean_list:
        # print("item:", item.step, item.value)
        step_list.append(item.step)
        value_list.append(item.value)

    # pick model of best performance
    step_list = np.array(step_list)
    value_list = np.array(value_list)
    argmax = np.argmax(value_list)
    max_step = step_list[argmax]
    max_value = value_list[argmax]

    return max_step, max_value


def generate_yaml(cfg, config_file, best_model):

    ft_cfg = cfg.copy()
    ft_cfg["OUTPUT_DIR"] = cfg["OUTPUT_DIR"][:-1] + "_ft/"
    ft_cfg["MODEL"]["WEIGHT"] = best_model
    ft_cfg["SOLVER"]["BASE_LR"] = 0.000002
    ft_cfg["DATASETS"]["TRAIN"] = '("RRPN_train_ft",)'

    ft_config_file = config_file[:-5] + "_ft.yaml"
    with open(ft_config_file, "w") as f:
        yaml.dump(ft_cfg, f)

    return ft_cfg, ft_config_file


def train_exec(config_file):
    command = "python3 tools/train_net.py --config {} --resume {}".format(config_file, False)
    print(os.system(command))



def ft_exec_with_best(config_file):

    cfg = yaml.load(open(config_file, "r"))
    print("cfg", cfg["OUTPUT_DIR"])

    dir_split = cfg["OUTPUT_DIR"].split("/")
    tensorboard_dir = os.path.join(
        "./tensorboard/",
        dir_split[-1] if len(dir_split[-1]) > 0 else dir_split[-2])

    tb_file = os.listdir(tensorboard_dir)[0]
    tb_path = os.path.join(tensorboard_dir, tb_file)

    max_step, max_value = pick_best(tb_path)

    best_model = os.path.join(
        cfg["OUTPUT_DIR"],
        "model_" + str(max_step).rjust(7, "0") + ".pth"
    )

    ft_cfg, ft_config_file = generate_yaml(cfg, config_file, best_model)

    print("Best performance at step", max_step, max_value)
    print("Now finetune with best model", best_model, "...")

    command = "python3 tools/train_net.py --config {} --resume {}".format(ft_config_file, "False")
    print("Go with", command)
    print(os.system(command))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--finetune",
        default=False,
        metavar="FILE",
        help="path to config file",
        type=bool,
    )

    args = parser.parse_args()

    assert os.path.isfile(args.config_file), "Configure file doesn't exist..."
    # config_file = "configs/arpn/e2e_rrpn_R_50_C4_1x_train_AFPN_RT.yaml"
    train_exec(args.config_file)
    if args.finetune:
        ft_exec_with_best(args.config_file)
