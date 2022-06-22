import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
import argparse
import json
import pprint
import signal
import socket
import time
from easydict import EasyDict
import yaml
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from calc_mAP import run_evaluation
from datasets import ava, spatial_transforms, temporal_transforms
from distributed_utils import init_distributed
import losses
from losses import *
from models import AVA_model
from scheduler import get_scheduler
from utils import *

import tensorboard_funcs
import wandb

def log_artifacts():
    results = wandb.Artifact(name = "results", type= "results", description = "results and config files")
    ckpts = wandb.Artifact(name = "models", type = "models", description= "model checkpoints")
    results.add_dir("./output/text")
    ckpts.add_dir("./output/saved_models")
    wandb.log_artifact(results)
    wandb.log_artifact(ckpts)
    wandb.finish()


class MyError(Exception): 
    pass


def handler(signum, frame):
    print("Trying to log artifacts.... (Ctrl-C to stop)")
    log_artifacts()
    raise MyError('Received signal ' + str(signum) +
                  ' on line ' + str(frame.f_lineno) +
                  ' in ' + frame.f_code.co_filename)    

def main(local_rank, args):
    """

    Main process code that controls one node in a parallel computation.

    rank and world size explained:
    https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank
    
    Parameters
    ----------
    local_rank : int
        variable generated by pytorch
        https://pytorch.org/docs/stable/elastic/multiprocessing.html
    args : argparse.Namespace
        args to pass down
    """
    rank, world_size = init_distributed(local_rank, args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # opt as in options are loaded from config files provided as argument.
    opt = EasyDict(config)
    opt.world_size = world_size

    
    
    if rank == 0:
        if not opt.experiment_name:
            raise ValueError("No experiment name specified in run config.")
        else:
            # resume does not take care of loading ckpts, you must do that yourself in the config for now.
            wandb.init(project='acar', name = opt.experiment_name, sync_tensorboard=True, resume = args.resume)
        mkdir(opt.result_path)
        mkdir(os.path.join(opt.result_path, 'tmp'))
        with open(os.path.join(opt.result_path, f'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, indent=2)
        logger = create_logger(os.path.join(opt.result_path, f'log.txt'))
        logger.info('opt: {}'.format(pprint.pformat(opt, indent=2)))
        
        writer = SummaryWriter(opt.tensorboard_path)
    else:
        logger = writer = None
    
    # synchronize all processes by blocking until all processes enter this func
    dist.barrier()
    
    # if need to set a seed, it'll be in the config
    random_seed(opt.manual_seed)
    # setting benchmark to True causes OOM in some cases
    if opt.get('cudnn', None) is not None:
        torch.backends.cudnn.deterministic = opt.cudnn.get('deterministic', False)
        torch.backends.cudnn.benchmark = opt.cudnn.get('benchmark', False)
    

    
    # create model from model config
    net = AVA_model(opt.model)
    net.cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank], broadcast_buffers=False)
    
    if rank == 0:
        # the node with rank 0 is responsible for logging these parameters
        logger.info(net)
        logger.info(parameters_string(net))

    if not opt.get('evaluate', False):
        # if this is a training run

        # train_aug is a dict of configs for augmentation pipeline
        train_aug = opt.train.augmentation
        
        spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in train_aug.spatial]
        spatial_transform = spatial_transforms.Compose(spatial_transform)

        temporal_transform = getattr(temporal_transforms, train_aug.temporal.type)(**train_aug.temporal.get('kwargs', {}))

        if opt.get('dataset', "ava") == "road":
            # ava is the default dataset when dataset is unspecified 
            # augmented train data has size 76139, the most of which have 91 frames. 
            # important to note that the actual images are not read into memory here
            train_data = ava.ROAD(
                opt.train.root_path,
                opt.train.annotation_path,
                opt.train.class_idx_path,
                "train_1",
                spatial_transform,
                temporal_transform,
            )
        else:
            train_data = ava.AVA(
                opt.train.root_path,
                opt.train.annotation_path,
                spatial_transform,
                temporal_transform,
            )

        # Distributed sampler gives each node a different set of sample
        # indices when its iterator is called.
        train_sampler = DistributedSampler(train_data, round_down=True)

        # AVADataLoader is a torch data loader
        train_loader = ava.AVADataLoader(
            train_data,
            batch_size=opt.train.batch_size,
            shuffle=False,
            num_workers=opt.train.get('workers', 1),
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
        
        if rank == 0:
            logger.info('# train data: {}'.format(len(train_data)))
            logger.info('train spatial aug: {}'.format(spatial_transform))
            logger.info('train temporal aug: {}'.format(temporal_transform))

            train_logger = Logger(
                os.path.join(opt.result_path, f'train.log'),
                ['epoch', 'loss', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, f'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss', 'lr'])
        else:
            train_logger = train_batch_logger = None

        optim_opt = opt.train.optimizer
        sched_opt = opt.train.scheduler

        optimizer = getattr(optim, optim_opt.type)(
            net.parameters(),
            lr=sched_opt.base_lr,
            **optim_opt.kwargs
        )
        scheduler = get_scheduler(sched_opt, optimizer, opt.train.n_epochs, len(train_loader))

    val_aug = opt.val.augmentation

    transform_choices, total_choices = [], 1
    for aug in val_aug.spatial:
        kwargs_list = aug.get('kwargs', {})
        if not isinstance(kwargs_list, list):
            kwargs_list = [kwargs_list]
        cur_choices = [getattr(spatial_transforms, aug.type)(**kwargs) for kwargs in kwargs_list]
        transform_choices.append(cur_choices)
        total_choices *= len(cur_choices)

    spatial_transform = []
    for choice_idx in range(total_choices):
        idx, transform = choice_idx, []
        for cur_choices in transform_choices:
            n_choices = len(cur_choices)
            cur_idx = idx % n_choices
            transform.append(cur_choices[cur_idx])
            idx = idx // n_choices
        spatial_transform.append(spatial_transforms.Compose(transform))

    temporal_transform = getattr(temporal_transforms, val_aug.temporal.type)(**val_aug.temporal.get('kwargs', {}))

    if opt.get('dataset', "ava") == "road":                                                    
        val_data = ava.ROADmulticrop(
            opt.val.root_path,
            opt.val.annotation_path,
            opt.val.class_idx_path,
            "val_1",
            spatial_transform,
            temporal_transform,
        )
    else:
        val_data = ava.AVAmulticrop(
            opt.val.root_path,
            opt.val.annotation_path,
            spatial_transform,
            temporal_transform,
        )


    val_sampler = DistributedSampler(val_data, round_down=False)

    val_loader = ava.AVAmulticropDataLoader(
        val_data,
        batch_size=opt.val.batch_size,
        shuffle=False,
        num_workers=opt.val.get('workers', 1),
        pin_memory=True,
        sampler=val_sampler
    )
    
    val_logger = None
    if rank == 0:
        logger.info('# val data: {}'.format(len(val_data)))
        logger.info('val spatial aug: {}'.format(spatial_transform))
        logger.info('val temporal aug: {}'.format(temporal_transform))

        val_log_items = ['epoch']
        if opt.val.with_label:
            val_log_items.append('loss')
        if opt.val.get('eval_mAP', None) is not None:
            val_log_items.append('mAP')
        if len(val_log_items) > 1:
            val_logger = Logger(
                os.path.join(opt.result_path, f'val.log'),
                val_log_items)
    
    if opt.get('pretrain', None) is not None:
        load_pretrain(opt.pretrain, net)
    
    begin_epoch = 1
    if opt.get('resume_path', None) is not None:
        if not os.path.isfile(opt.resume_path):
            opt.resume_path = os.path.join(opt.model_path, opt.resume_path)
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage.cuda())

        begin_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            logger.info('Resumed from checkpoint {}'.format(opt.resume_path))

        if not opt.get('evaluate', False):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                logger.info('Also loaded optimizer and scheduler from checkpoint {}'.format(opt.resume_path))

    elif opt.loss.type == "wBCE":
        criterion = wBCE(train_data.action_counts, train_data.total_boxes)
        act_func = torch.sigmoid
    else:
        criterion, act_func = getattr(losses, opt.loss.type)(**opt.loss.get('kwargs', {}))

                        ###################################
                            # TRAINING AND VALIDATION
                        ###################################

    if opt.get('evaluate', False):  # evaluation mode
        val_epoch(begin_epoch - 1, val_loader, net, criterion, act_func,
                  opt, logger, val_logger, rank, world_size, writer)
    else:  # training and validation mode
        for e in range(begin_epoch, opt.train.n_epochs + 1):
            train_sampler.set_epoch(e)
            train_epoch(e, train_loader, net, criterion, act_func, optimizer, scheduler,
                        opt, logger, train_logger, train_batch_logger, rank, world_size, writer)
            
            if e % opt.train.val_freq == 0:
                val_epoch(e, val_loader, net, criterion, act_func,
                          opt, logger, val_logger, rank, world_size, writer)

    if rank == 0:
        log_artifacts()
        writer.close()
    
    
def train_epoch(epoch, data_loader, model, criterion, act_func, optimizer, scheduler, 
                opt, logger, epoch_logger, batch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Training at epoch {}'.format(epoch))
    
    # sets model to train mode. This is a torch method.
    model.train()
    
    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)
    loss_time = AverageMeter(opt.print_freq)
    losses = AverageMeter(opt.print_freq)
    global_losses = AverageMeter()
    
    train_epoch_pred_prob = torch.tensor([]).cpu()
    train_epoch_targets = torch.tensor([]).cpu()

    end_time = time.time()
    for i, data in enumerate(data_loader):
        # data should now contain the actual data.

        """
        data:
        - clips:
            ex. torch.Size([8, 3, 32, 256, 341])
            representing (batch size, rgb channels, number of continuous frames (for slowfast), height, width)
        - aug_info
            List of augmentations performed on each individual datapoint
        - filenames
            which (ex. of the 22 ROAD dataset videos) each sample came from.
        - labels
            List of {'bounding_box': [...], 'label: [...]}
        - mid_times
            The times in the videos that the frames are taken.
        """

        data_time.update(time.time() - end_time)

        curr_step = (epoch - 1) * len(data_loader) + i
        scheduler.step(curr_step)

        # FORWARD PASS !!!!
        ret = model(data)

        num_rois = ret['num_rois']
        outputs = ret['outputs']
        targets = ret['targets']
        
        tot_rois = torch.Tensor([num_rois]).cuda()
        dist.all_reduce(tot_rois)
        tot_rois = tot_rois.item()

        if tot_rois == 0:
            end_time = time.time()
            continue

        optimizer.zero_grad()

        if num_rois > 0:
            loss = criterion(outputs, targets)
            loss = loss * num_rois / tot_rois * world_size
            batch_pred_prob = act_func(outputs)
            train_epoch_pred_prob = torch.cat( (train_epoch_pred_prob, batch_pred_prob.detach().cpu()), axis=0 )
            train_epoch_targets = torch.cat( (train_epoch_targets, targets.detach().cpu()), axis=0 )
        else:
            loss = torch.tensor(0).float().cuda()
            for param in model.parameters():
                if param.requires_grad:
                    loss = loss + param.sum()
            loss = 0. * loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.train.max_norm)
        optimizer.step()

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item(), tot_rois)
        global_losses.update(reduced_loss.item(), tot_rois)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.print_freq == 0 and rank == 0:
            writer.add_scalar('train/loss', losses.avg, curr_step + 1)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], curr_step + 1)
            
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': curr_step + 1,
                'loss': losses.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                            epoch,
                            i + 1,
                            len(data_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses))

    if rank == 0:
        writer.add_scalar('train/epoch_loss', global_losses.avg, epoch)
        train_class_list = [classification + "_train" for classification in opt.class_list]
        tensorboard_funcs.add_pr_curves_to_tensorboard(train_epoch_targets, train_epoch_pred_prob, train_class_list, writer, epoch)
        writer.flush()

        epoch_logger.log({
            'epoch': epoch,
            'loss': global_losses.avg,
            'lr': optimizer.param_groups[0]['lr']
        })

        logger.info('-' * 100)
        logger.info(
            'Epoch [{}/{}]\t'
            'Loss {:.4f}'.format(
                epoch,
                opt.train.n_epochs,
                global_losses.avg))

        if epoch % opt.train.save_freq == 0:
            save_file_path = os.path.join(opt.model_path, 'ckpt_{}.pth.tar'.format(epoch))
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(states, save_file_path)
            logger.info('Checkpoint saved to {}'.format(save_file_path))
        
        logger.info('-' * 100)
        
        tensorboard_funcs.add_model_weights_as_histogram(model, writer, epoch)            

def val_epoch(epoch, data_loader, model, criterion, act_func,
              opt, logger, epoch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Evaluation at epoch {}'.format(epoch))

    model.eval()
    
    calc_loss = opt.val.with_label
    save_pred = (opt.val.get('eval_mAP', None) is not None)
    if save_pred:
        out_file = open(os.path.join(opt.result_path, 'tmp', f'predict_rank_{rank}.csv'), 'w')
    
    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)
    if calc_loss:
        global_losses = AverageMeter()
        
    val_epoch_pred_prob = torch.tensor([]).cpu()
    val_epoch_targets = torch.tensor([]).cpu()

    end_time = time.time()
    for i, data in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        with torch.no_grad():
            ret = model(data, evaluate=True)
            num_rois = ret['num_rois']
            outputs = ret['outputs']
            targets = ret['targets']
        if num_rois == 0:
            end_time = time.time()
            continue
        
        batch_pred_prob = act_func(outputs)
        val_epoch_pred_prob = torch.cat( (val_epoch_pred_prob, batch_pred_prob.detach().cpu()), axis=0 )
        val_epoch_targets = torch.cat( (val_epoch_targets, targets.detach().cpu()), axis=0 )


        if calc_loss:
            loss = criterion(outputs, targets)
            global_losses.update(loss.item(), num_rois)

        if save_pred:
            fnames, mid_times, bboxes = ret['filenames'], ret['mid_times'], ret['bboxes']
            outputs = act_func(outputs).cpu().data
            idx_to_class = data_loader.dataset.idx_to_class
            for k in range(num_rois):
                prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f"%(fnames[k], mid_times[k],
                                                      bboxes[k][0], bboxes[k][1],
                                                      bboxes[k][2], bboxes[k][3])
                for cls in range(outputs.shape[1]):
                    score_str = '%.3f'%outputs[k][cls]
                    # Hack: Trained with too many classes, ignore anything out of range
                    if cls >= len(idx_to_class):
                        break 
                    out_file.write(prefix + ",%d,%s\n" % (idx_to_class[cls]['id'], score_str))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.print_freq == 0 and rank == 0:
            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                            epoch,
                            i + 1,
                            len(data_loader),
                            batch_time=batch_time,
                            data_time=data_time))
    
    if calc_loss:
        total_num = torch.Tensor([global_losses.count]).cuda()
        loss_sum = torch.Tensor([global_losses.avg * global_losses.count]).cuda()
        dist.all_reduce(total_num)
        dist.all_reduce(loss_sum)
        final_loss = loss_sum.item() / (total_num.item() + 1e-10)

    if save_pred:
        out_file.close()
        dist.barrier()
        
    if rank == 0:
        val_log = {'epoch': epoch}
        val_str = 'Epoch [{}]'.format(epoch)

        if calc_loss:
            writer.add_scalar('val/epoch_loss', final_loss, epoch)
            val_log['loss'] = final_loss
            val_str += '\tLoss {:.4f}'.format(final_loss)

        if save_pred:
            result_file = os.path.join(opt.result_path, f'predict_epoch_{epoch}.csv')
            with open(result_file, 'w') as of:
                for r in range(world_size):
                    with open(os.path.join(opt.result_path, 'tmp', f'predict_rank_{r}.csv'), 'r') as f:
                        of.writelines(f.readlines())

            eval_mAP = opt.val.eval_mAP
            metrics = run_evaluation(
                open(eval_mAP.labelmap, 'r'), 
                open(eval_mAP.groundtruth, 'r'),
                open(result_file, 'r'),
                open(eval_mAP.exclusions, 'r') if eval_mAP.get('exclusions', None) is not None else None, 
                logger
            )

            mAP = metrics['PascalBoxes_Precision/mAP@0.5IOU']
            writer.add_scalar('val/mAP', mAP, epoch)
            val_log['mAP'] = mAP
            val_str += '\tmAP {:.6f}'.format(mAP)
        
        val_class_list = [classification + "_val" for classification in opt.class_list]
        tensorboard_funcs.add_pr_curves_to_tensorboard(val_epoch_targets, val_epoch_pred_prob, val_class_list, writer, epoch)
        
        writer.flush()
        
        if calc_loss or save_pred:
            epoch_logger.log(val_log)

            logger.info('-' * 100)
            logger.info(val_str)
            logger.info('-' * 100)

    dist.barrier()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch AVA Training and Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--nproc_per_node', type=int, default=8)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--master_port', type=int, default=31114)
    parser.add_argument('--nnodes', type=int, default=None)
    parser.add_argument('--node_rank', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    opt = EasyDict(config)
    args.nproc_per_node = opt.nproc_per_node # hack to make the rest of the code work
    # if we're debugging, don't log anything

    ####################################################
    # To turn off wandb, run: export WANDB_MODE=offline
    ####################################################

    torch.multiprocessing.spawn(main, args=(args,), nprocs=opt.nproc_per_node)