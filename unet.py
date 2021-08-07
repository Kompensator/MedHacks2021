from random import Random
from termcolor import cprint
import yaml
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import nibabel as nib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet_datasets import *
from unet_utils import *
from unet_model import *
import pickle
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from mlflow import start_run, log_metric, log_param

class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        sample_plotter (callable): saves sample inputs, network outputs and targets to a given directory
            during validation phase
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, sample_plotter=None,
                 skip_train_validation=False, validate_after_epochs=1,
                 log_after_epochs=20, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.validate_after_epochs = validate_after_epochs
        self.log_after_epochs = log_after_epochs
        self.validate_now = False
        self.log_now = False

        cprint(f'eval_score_higher_is_better: {eval_score_higher_is_better}', 'green')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
        
        os.chdir(r"C:\Users\dingyi.zhang\Documents\CV-Calcium-DY")
        # if checkpoint_dir == 'none':
        #     self.writer = SummaryWriter()
        # else:
        #     self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        # self.writer = SummaryWriter()

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.sample_plotter = sample_plotter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

    @classmethod
    def from_checkpoint(cls, resume, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, **kwargs):
        cprint(f"Loading checkpoint '{resume}'...", 'green')
        state = load_checkpoint(resume, model, optimizer)
        cprint(f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}", 'green')
        checkpoint_dir = os.path.split(resume)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   skip_train_validation=state.get('skip_train_validation', False),
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=int(1e5),
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, sample_plotter=None,
                        skip_train_validation=False, **kwargs):
        cprint(f"Logging pre-trained model from '{pre_trained}'...", 'green')
        load_checkpoint(pre_trained, model, None)
        if 'checkpoint_dir' not in kwargs:
            checkpoint_dir = os.path.split(pre_trained)[0]
        else:
            checkpoint_dir = kwargs.pop('checkpoint_dir')
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation)

    def fit(self, run_name="default"):
        with start_run(run_name=run_name):
            self.run_name = run_name
            save_cp_interval = 2
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.DICE = GeneralizedDiceLoss()
            for _ in range(self.num_epoch, 200):
                try:
                    should_terminate = self.train()
                except KeyboardInterrupt:           # for some reason this try except loop does not work
                    cprint("Training stopped manually, saving model...", "yellow")
                    self.save_model()

                if should_terminate:
                    cprint('Stopping criterion is satisfied. Finishing training', 'yellow')
                    self.save_model()
                    return
                
                if _ % save_cp_interval == 0:
                    self.save_model()
                self.num_epoch += 1
        
            cprint(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...", 'yellow')
            self.save_model()


    def save_model(self):
        torch.save(self.model.state_dict(), 'checkpoints\\{}_{}.h5'.format(self.run_name, self.num_epoch))

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()

        self.model.train()
        self.num_iterations = 0
        dice_train = []

        for t in self.loaders['train']:
            input, target, weight = self._split_training_batch(t)
            iteration_per_epoch = self.loaders['train'].dataset.__len__() // input.shape[0] 
            
            with torch.cuda.amp.autocast(enabled=True):
                dice, loss = self._forward_pass(input, target, weight)

            dice_train.append(dice)
            cprint(f'Training iteration [{self.num_iterations + 1}/{iteration_per_epoch}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}] Loss [{loss}]', 'green')
    
            train_losses.update(loss.item(), self._batch_size(input))

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # if self.should_stop():
            #     return True
            self.num_iterations += 1
        
        dice_train_avg = sum(dice_train) / len(dice_train)
        self.model.eval()
        val_loss, eval_score = self.validate()
        self.model.train()

        # FIXME LR scheduler is being annoying
        self.scheduler.step(eval_score)

        # is_best = self._is_best_eval_score(eval_score)

        # self._save_checkpoint(is_best)

        # log stats, params and images
        cprint(f'Train loss: {train_losses.avg}. Validation loss: {val_loss}', 'blue')
        cprint(f'Train DICE: {dice_train_avg}. Validation DICE: {eval_score}', 'blue')
        self._log_stats(train_losses.avg, val_loss, dice_train_avg, eval_score)
        self._log_lr()
        # self._log_params()
        # self._log_images(input, target, output, 'train_')

        if self.should_stop():
            return True
        return False

    def should_stop(self):
        """
        Training will terminate if learning rate drops below a threshold
        """
        # min_lr = 1e-6
        # lr = self.optimizer.param_groups[0]['lr']
        # if lr < min_lr:
        #     cprint(f'Learning rate below the minimum {min_lr}.', 'red')
        #     return True

        return False

    def validate(self):
        val_losses = RunningAverage()
        dice_val = []

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):

                input, target, weight = self._split_training_batch(t)

                dice, loss = self._forward_pass(input, target, weight)
                val_losses.update(loss.item(), self._batch_size(input))
                dice_val.append(dice)

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                # if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #     output = self.model.final_activation(output)

                # if i % 100 == 0:
                #     self._log_images(input, target, output, 'val_')

                # eval_score = self.eval_criterion(output, target)          # chance this to dice
                # val_scores.update(eval_score.item(), self._batch_size(input))

                # if self.sample_plotter is not None:
                #     self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break
            
            # self._log_stats('val', val_losses.avg, val_scores.avg)
            # cprint(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {0}', 'blue')
            return val_losses.avg, sum(dice_val)/len(dice_val)

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None       # FIXME why hard code weight=None?
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)
        
        dice = self.DICE(output, target)
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target.long())
        else:
            loss = self.loss_criterion(output, target.long(), weight)

        return dice.item(), loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            cprint(f'Saving new best evaluation metric: {eval_score}', 'yellow')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=None)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        # self.writer.add_scalar('learning_rate', lr, self.num_epoch)
        log_metric("lr", lr, self.num_epoch)

    def _log_stats(self, loss_avg, val_loss, train_dice, val_dice):
        log_metric('train loss', loss_avg, self.num_epoch)
        log_metric('val loss', val_loss, self.num_epoch)
        log_metric('train DICE', train_dice, self.num_epoch)
        log_metric('val DICE', val_dice, self.num_epoch)

    def _log_params(self):
        # cprint('Logging model parameters and gradients', 'yellow')
        for name, value in self.model.named_parameters():
            pass
            # self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_epoch)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_epoch)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                pass
                # self.writer.add_image(prefix + tag, image, self.num_epoch, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)


def build_trainer(config):
    # equivalent to trainer.3DUnetTrainerBuilder.build()
    model_config = config['model']
    model_name = model_config['name']
    model = None
    if model_name == "UNet3D":
        model = UNet3D(**model_config)
        with open("UNet3D_config", "wb") as f:
            pickle.dump(model_config, f)
            cprint("Saved model config for UNet3D", "green")
    elif model_name == "ResidualUNet3D":
        model = ResidualUNet3D(**model_config)
        with open("ResUNet3D_config", "wb") as f:
            cprint("Saved model config for Residual UNet3D", "green")
            pickle.dump(model_config, f)
    else:
        cprint(f"{model_name}type not found", "red")
    assert model is not None, "Model not created"

    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        cprint(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    cprint(f"Sending the model to '{config['device']}'", 'green')
    model = model.to(device)

    cprint(f'Number of learnable params {get_number_of_learnable_parameters(model)}', 'green')

    loss_criterion = get_loss_criterion(config)
    eval_criterion = get_evaluation_metric(config)

    train_dataset = AlphaTau3_train(start=0.0, end=0.01)
    val_dataset = AlphaTau3_train(start=0.01, end=0.012)

    train_loader = DataLoader(train_dataset, batch_size=1)       #NOTE: batchsize is here!
    val_loader = DataLoader(val_dataset, batch_size=1)
    train_loaders = {"train": train_loader, "val": val_loader}

    optimizer = create_optimizer(config['optimizer'], model)        # NOTE opti is Adam
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    # Create model trainer
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']
    # status
    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    tensorboard_formatter = TensorboardFormatter()
    sample_plotter = None

    if resume is not None:
        # resume from checkpoint
        return UNet3DTrainer.from_checkpoint(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             loaders=train_loaders,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             **trainer_config)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             device=config['device'],
                                             loaders=train_loaders,
                                             **trainer_config)
    else:
        # start training from scratch
        return UNet3DTrainer(model=model,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             loss_criterion=loss_criterion,
                             eval_criterion=eval_criterion,
                             device=config['device'],
                             loaders=train_loaders,
                             tensorboard_formatter=tensorboard_formatter,
                             sample_plotter=sample_plotter,
                             **trainer_config)


def main(run_name='default'):
    config = get_config()
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        cprint(f'Seed the RNG for all devices with {manual_seed}', 'green')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    trainer = build_trainer(config)
    # Start training
    trainer.fit(run_name=run_name)


if __name__ == '__main__':
    main(run_name='dim512_features9_10%')