import sys
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_mami import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader

from scheduler import create_scheduler
from optim import create_optimizer
from sklearn import metrics


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image_id, image, text, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, labels = image.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(image, text_input, labels, alpha=alpha, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, output_dir=None):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    image_id_list = []
    labels_list = []
    pred_labels_list = []
    pred_probas_list = []

    for image_id, image, text, labels in metric_logger.log_every(data_loader, print_freq, header):
        image, labels = image.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        logits = model(image, text_input, train=False)

        argmax_accuracy = calculate_argmax_accuracy(logits, labels)

        metric_logger.meters['argmax_acc'].update(argmax_accuracy.item(), n=image.size(0))

        pred_logits = nn.Sigmoid()(logits)
        accuracy = (pred_logits.round() == labels).sum() / labels.size(0)

        image_id_list += image_id
        labels_list += labels.int().tolist()
        pred_labels_list += pred_logits.round().int().tolist()
        pred_probas_list += pred_logits.tolist()

        metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    if output_dir is not None:
        save_path = os.path.join(output_dir, 'test_pred.jsonl')
        with open(save_path, 'w') as out_f:
            for gold, pred, prob in zip(labels_list, pred_labels_list, pred_probas_list):
                data = {}
                data['mis'] = gold[0]
                data['sha'] = gold[1]
                data['ste'] = gold[2]
                data['obj'] = gold[3]
                data['vio'] = gold[4]
                data['oth'] = gold[5]

                data['mis_pred'] = pred[0]
                data['sha_pred'] = pred[1]
                data['ste_pred'] = pred[2]
                data['obj_pred'] = pred[3]
                data['vio_pred'] = pred[4]
                data['oth_pred'] = pred[5]

                data['mis_prob'] = prob[0]
                data['sha_prob'] = prob[1]
                data['ste_prob'] = prob[2]
                data['obj_prob'] = prob[3]
                data['vio_prob'] = prob[4]
                data['oth_prob'] = prob[5]

                out_f.write(json.dumps(data) + '\n')

        # with open('output/mami/internal_test/answer.txt', 'w') as out_f:
        #     for id, pred in zip(image_id_list, pred_labels_list):
        #         out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(id, pred[0], pred[1], pred[2], pred[3], pred[4]))
        #
        # with open('output/mami/internal_test/answer_prob.txt', 'w') as out_f:
        #     for id, prob in zip(image_id_list, pred_probas_list):
        #         out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(id, prob[0], prob[1], prob[2], prob[3], prob[4]))
        #
        # with open('output/mami/test_pred_cxpy.jsonl', 'w') as cxpy_out_f:
        #     for gold, pred, prob in zip(labels_list, pred_labels_list, pred_probas_list):
        #         data = {}
        #         data['input'] = {'bin_label_id': gold[0]}
        #         data['pred_conf_threshold'] = 0.5
        #         data['pred_score'] = prob[0]
        #         data['pred_label_id'] = pred[0]
        #
        #         cxpy_out_f.write(json.dumps(data) + '\n')

        mis_f1, multilabel_f1 = calculate_multilabel_f1(save_path)
        print('Mis F1: {} | Multi-label F1: {}'.format(mis_f1, multilabel_f1))

    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def calculate_argmax_accuracy(logits, labels):

    logits_argmax = logits.argmax(dim=-1).view(-1,1)
    preds = torch.zeros_like(logits)
    preds.scatter_(1, logits_argmax, 1)

    assert preds.shape == labels.shape

    scores = preds * labels

    argmax_accuracy = (scores.sum() / labels.shape[0])

    return argmax_accuracy


def calculate_multilabel_f1(file):

    sublabel_name_list = ['sha','ste','obj','vio']

    with open(file, 'r') as f:
        lines = f.readlines()
        line_list = [json.loads(line) for line in lines]
        df = pd.DataFrame(line_list)

        pred, gold = df['mis_pred'].tolist(), df['mis'].tolist()
        mis_f1 = calculate_f1(pred, gold)

        results = []
        total_occurences = 0

        for name in sublabel_name_list:
            pred, gold = df['{}_pred'.format(name)].tolist(), df[name].tolist()
            f1_score = calculate_f1(pred, gold)
            weight = gold.count(True)
            total_occurences += weight

            results.append(f1_score * weight)

        multilabel_f1 = sum(results) / total_occurences

    return mis_f1, multilabel_f1

def calculate_f1(pred_values, gold_values):
    matrix = metrics.confusion_matrix(gold_values, pred_values)
    matrix = check_matrix(matrix, gold_values, pred_values)

    # positive label
    if matrix[0][0] == 0:
        pos_precision = 0.0
        pos_recall = 0.0
    else:
        pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

    if (pos_precision + pos_recall) != 0:
        pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    else:
        pos_F1 = 0.0

    # negative label
    neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    if neg_matrix[0][0] == 0:
        neg_precision = 0.0
        neg_recall = 0.0
    else:
        neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
        neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

    if (neg_precision + neg_recall) != 0:
        neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    else:
        neg_F1 = 0.0

    f1 = (pos_F1 + neg_F1) / 2
    return f1

def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix


@torch.no_grad()
def official_test_evaluate(model, data_loader, tokenizer, device, config, output_dir=None):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    image_id_list = []
    pred_labels_list = []
    pred_probas_list = []

    for image_id, image, text, labels in metric_logger.log_every(data_loader, print_freq, header):
        image, labels = image.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        logits = model(image, text_input, train=False)
        pred_logits = nn.Sigmoid()(logits)

        image_id_list += image_id
        pred_labels_list += pred_logits.round().int().tolist()
        pred_probas_list += pred_logits.tolist()

    if output_dir is not None:
        with open(os.path.join(output_dir, 'answer.txt'), 'w') as out_f:
            for id, pred in zip(image_id_list, pred_labels_list):
                out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(id, pred[0], pred[1], pred[2], pred[3], pred[4]))

        with open(os.path.join(output_dir, 'answer_prob.txt'), 'w') as out_f:
            for id, prob in zip(image_id_list, pred_probas_list):
                out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(id, prob[0], prob[1], prob[2], prob[3], prob[4]))

        with open(os.path.join(output_dir, 'answer_argmax.txt'), 'w') as out_f:
            for id, pred, prob in zip(image_id_list, pred_labels_list, pred_probas_list):

                sublabel_pred = pred[1:5]
                argmax_sublabel_pred = [0] * len(sublabel_pred)
                argmax_sublabel_pred[np.argmax(sublabel_pred)] = 1

                out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(id, pred[0], argmax_sublabel_pred[0],
                                                                    argmax_sublabel_pred[1], argmax_sublabel_pred[2],
                                                                    argmax_sublabel_pred[3]))


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    
    #### Dataset #### 
    print("Creating mami datasets")
    datasets = create_dataset('mami', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None, None]
    
    train_loader, val_loader, test_loader, off_test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train']]+[config['batch_size_test']]*3,
                                              num_workers=[4,4,4,4],is_trains=[True, False, False, False],
                                              collate_fns=[None, None, None ,None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
        
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
                
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    print("Start training")
    start_time = time.time()

    best = 0
    best_epoch = 0

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        test_stats = evaluate(model, test_loader, tokenizer, device, config, args.output_dir)

        if args.off_test_evaluate:
            official_test_evaluate(model, off_test_loader, tokenizer, device, config, args.output_dir)

        if utils.is_main_process():
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats['argmax_acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['argmax_acc'])
                    best_epoch = epoch

        if args.evaluate:
            break
        dist.barrier()
                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/MAMI.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/mami')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--off_test_evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)