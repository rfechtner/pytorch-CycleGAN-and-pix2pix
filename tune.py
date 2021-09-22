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
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from sklearn.utils import Bunch
import shlex
from ray import tune
from metrics import peak_based_f1
from util import util
from logger import CustomTBXLogger
from util.visualizer import prediction2fig

from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB


def train(config):
    logger = CustomTBXLogger(
        config=config,
        logdir=tune.get_trial_dir()
    )

    logger.init()

    opt_str = " ".join(["--{k} {v}".format(k=key, v=val) for key, val in config["train"].items()])
    opt_str += " --name rt_{}_{}".format(config["other"]["name_prefix"], os.path.basename(tune.get_trial_dir()))

    opt = TrainOptions().parse(opt_str=opt_str)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    val_dataset = create_dataset(opt, phase="val")
    val_dataset_size = len(val_dataset)
    print('The number of validation images = %d' % val_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % config['val']['metric_freq'] == 0:
            print('running validation at the end of epoch %d' % (epoch))
            for i, data in enumerate(val_dataset):
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths

                print(img_path)

                source = util.tensor2im(visuals['real_A'])
                true = util.tensor2im(visuals['real_B'])
                pred = util.tensor2im(visuals['fake_B'])

                ### Calculate validation score
                f1 = peak_based_f1(true, pred)

                if i == 0:
                    print("Logging image")
                    logger.log_figure("image", prediction2fig(source, true, pred, f1), step=epoch)
                tune.report(f1=f1['f1'])

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


if __name__ == '__main__':

    config = {
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
                "pool_size": 5000,
                ### Tuneable
                "ngf": tune.choice([16, 32, 64, 128]),
                "ndf": tune.choice([16, 32, 64, 128]),
                "netD": tune.choice(["basic", "n_layers", "pixel"]),
                "netG": tune.choice(["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"]),
                "n_layers_D": tune.choice([0, 3, 8]),
                "norm": tune.choice(["instance", "batch", "none"]),
                "batch_size": tune.choice([1, 8, 16]),
                "lr": tune.loguniform(1e-5, 1e-2),
                "gan_mode": tune.choice(["vanilla", "lsgan", "wgangp"]),
                "lr_policy": tune.choice(["linear", "step", "plateau", "cosine"]),
                "lambda_L1": tune.lograndint(1, 1000)
            },
            "val": {
                "metric_freq": 1
            }
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=4,
        stop_last_trials=False
    )

    bohb_search = TuneBOHB(
        max_concurrent=4
    )

    analysis = tune.run(
        train,
        name="bf_tune",
        config=config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        metric="f1",
        mode="max",
        stop={"training_iteration": 150},
        resources_per_trial={"cpu": 16, "gpu": 1},
        local_dir="/project/ag-pp2/13_ron/masterthesis_workingdir/Trainings/pix2pix/ray_tune",
        verbose=2
    )


    print("Best hyperparameters found were: ", analysis.best_config)