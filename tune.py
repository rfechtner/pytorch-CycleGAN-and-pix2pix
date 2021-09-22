"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os.path
import time
import json
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

import torch
from ray import tune
from metrics import peak_based_f1
from util import util
from logger import CustomTBXLogger
from util.visualizer import prediction2fig

from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import ray

from skimage.metrics import normalized_root_mse
from sklearn.metrics import jaccard_score

def train(config, checkpoint_dir=None, fixed_config=None):
    logger = CustomTBXLogger(
        config=config,
        logdir=tune.get_trial_dir()
    )

    logger.init()

    if fixed_config is not None:
        config.update(fixed_config["train"])
        print(config)

    opt_strs = []
    for key, val in config.items():
        if not isinstance(val, bool):
            opt_strs.append("--{k} {v}".format(k=key, v=val))
        else:
            if val:
                opt_strs.append("--{k}".format(k=key))

    opt_str = " ".join(opt_strs)
    opt_str += " --name rt_{}_{}".format(fixed_config["other"]["name_prefix"], tune.get_trial_id())

    #if checkpoint_dir:
    #    print("Checkpoint should be loaded from " + checkpoint_dir)
    #    opt_str += " --continue_train"

    opt = TrainOptions().parse(opt_str=opt_str)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    val_dataset = create_dataset(opt, phase="val")
    val_dataset_size = len(val_dataset)
    print('The number of validation images = %d' % val_dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    total_iters = 0  # the total number of training iterations

    # Override model loading
    if checkpoint_dir:
        print("loading checkpoint from file: " + checkpoint_dir)
        path = os.path.join(checkpoint_dir, "checkpoint")
        print(" -> " + str(path))
        with open(path) as f:
            state = json.loads(f.read())
            opt.epoch_count = state["step"] + 1
            total_iters += state["iter"] + 1

        for name in model.model_names:
            if isinstance(name, str):
                checkpoint = os.path.join(checkpoint_dir, "{}_checkpoint".format(name))
                print(" -> " + str(checkpoint))

                net = getattr(model, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                state_dict = torch.load(checkpoint, map_location=model.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if epoch % fixed_config['val']['metric_freq'] == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:

                path = os.path.join(checkpoint_dir, "checkpoint")
                print(" -> " + str(path))
                with open(path, "w") as f:
                    f.write(json.dumps({
                        "step": epoch + 1,
                        "iter": total_iters
                    }))

                for name in model.model_names:
                    path = os.path.join(checkpoint_dir, "{}_checkpoint".format(name))
                    print(" -> " + str(path))

                    if isinstance(name, str):
                        net = getattr(model, 'net' + name)

                        if len(model.gpu_ids) > 0 and torch.cuda.is_available():
                            torch.save(net.module.cpu().state_dict(), path)
                            net.cuda(model.gpu_ids[0])
                        else:
                            torch.save(net.cpu().state_dict(), path)

            print('running validation at the end of epoch %d' % (epoch))
            for i, data in enumerate(val_dataset):
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results

                source = util.tensor2im(visuals['real_A'])
                true = util.tensor2im(visuals['real_B'])
                pred = util.tensor2im(visuals['fake_B'])

                ### Calculate validation score
                f1 = peak_based_f1(true, pred)
                nrmse = normalized_root_mse(true, pred)
                iou = jaccard_score((true.flatten() > 0), (pred.flatten() > 0))

                if i == 0:
                    print("saving example image (f1 = {:.03f})".format(f1["f1"]))
                    logger.log_figure("image", prediction2fig(source, true, pred, f1), step=epoch)


                lr = model.schedulers[0].get_lr()

                tune.report(f1=f1['f1'], lr=lr, nrmse=nrmse, iou=iou)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


if __name__ == '__main__':
    ray.init(dashboard_host='0.0.0.0')

    tuneable_config = {
        "ngf": tune.choice([16, 32, 64, 128]),
        "ndf": tune.choice([16, 32, 64, 128]),
        "netD": tune.choice(["basic", "n_layers", "pixel"]),
        "netG": tune.choice(["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"]),
        "n_layers_D": tune.choice([0, 3, 5]),
        "norm": tune.choice(["instance", "batch", "none"]),
        "batch_size": tune.choice([1, 4, 8]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "gan_mode": tune.choice(["vanilla", "lsgan", "wgangp"]),
        "lr_policy": tune.choice(["linear", "step", "plateau", "cosine"]),
        "lambda_L1": tune.lograndint(1, 1000)
    }

    fixed_config = {
            "other": {
                "name_prefix": "BF2PSIVA"
            },
            "train": {
                "model": "pix2pix",
                "input_nc": 1,
                "output_nc": 1,
                "max_dataset_size": 500,
                "dataroot": "/project/ag-pp2/13_ron/masterthesis_workingdir/Datasets/new/BF2FLAVG_cropped_4th_rnd3rd/",
                "n_epochs": 100,
                "n_epochs_decay": 50,
                "save_epoch_freq": 25,
                "pool_size": 2500,
                "display_id": 0,
                "num_threads": 16,
                "no_html": True,
                "checkpoints_dir": "/project/ag-pp2/13_ron/masterthesis_workingdir/Trainings/pix2pix/ray_tune/checkpoints"
            },
            "val": {
                "metric_freq": 5
            }
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=int(fixed_config["train"]["max_dataset_size"]) * (int(fixed_config["train"]["n_epochs"]) + int(fixed_config["train"]["n_epochs_decay"])),
        stop_last_trials=False
    )

    bohb_search = TuneBOHB(
        max_concurrent=5
    )

    analysis = tune.run(
        tune.with_parameters(train, fixed_config=fixed_config),
        name="bf_tune",
        config=tuneable_config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=2,
        metric="f1",
        mode="max",
        #stop={"training_iteration": 150},
        resources_per_trial={"cpu": 32, "gpu": 1},
        local_dir="/project/ag-pp2/13_ron/masterthesis_workingdir/Trainings/pix2pix/ray_tune",
        verbose=3
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    df = analysis.results_df
    df.to_csv("/project/ag-pp2/13_ron/masterthesis_workingdir/Trainings/pix2pix/ray_tune/bf_tune/analysis.csv")