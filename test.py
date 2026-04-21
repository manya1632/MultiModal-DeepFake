import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from ruamel.yaml import YAML
yaml = YAML()
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.HAMMER import HAMMER

import hashlib
from dataclasses import dataclass
from typing import Literal, Optional
from models.watermark_image_decoder import ImageWatermarkDecoder
from models.watermark_text_decoder import TextWatermarkDecoder
from utils.metrics import compute_nc
from integrity.model_integrity import verify_model, ModelIntegrityError

@dataclass
class DetectionResult:
    """Output of the inference pipeline for a single sample."""
    label: Literal["real", "fake"]
    trust_score: float          # 0.7 * hammer_score + 0.3 * watermark_score
    hammer_score: float         # P(fake) from HAMMER BIC head
    watermark_score: float      # mean NC of extracted watermarks (0.0 if invalid)
    watermark_valid: bool       # watermark_score >= 0.5
    bbox: Optional[list]        # [cx, cy, w, h] normalized, or None
    fake_token_positions: list  # token indices predicted as fake


def compute_trust_score(hammer_score: float, watermark_score: float) -> float:
    """Compute final trust score: 0.7 * hammer_score + 0.3 * watermark_score.
    Validates: Requirements 10.4"""
    return 0.7 * hammer_score + 0.3 * watermark_score


def extract_watermark_score(
    image: torch.Tensor,
    text_embeddings: torch.Tensor,
    wm_img_dec: ImageWatermarkDecoder,
    wm_txt_dec: TextWatermarkDecoder,
    expected_m_T: torch.Tensor,
    expected_m_I: torch.Tensor,
    device: torch.device,
) -> tuple:
    """Extract watermarks and compute consistency score.
    Returns (watermark_score, watermark_valid).
    On any error, returns (0.0, False) without raising.
    Validates: Requirements 6.1, 6.2, 6.3, 10.6"""
    try:
        if image is None or text_embeddings is None:
            return 0.0, False
        if image.ndim != 4 or image.shape[1] != 3:
            return 0.0, False
        if text_embeddings.ndim != 3:
            return 0.0, False
        if torch.isnan(image).any() or torch.isnan(text_embeddings).any():
            return 0.0, False

        with torch.no_grad():
            m_T_hat = wm_img_dec(image)           # (B, 128)
            m_I_hat = wm_txt_dec(text_embeddings)  # (B, 128)

        # Compute NC for each sample and average
        nc_img = compute_nc(expected_m_T.float(), (m_T_hat > 0.5).float())
        nc_txt = compute_nc(expected_m_I.float(), (m_I_hat > 0.5).float())
        watermark_score = float((nc_img + nc_txt) / 2.0)
        watermark_valid = watermark_score >= 0.5
        return watermark_score, watermark_valid
    except Exception:
        return 0.0, False


def load_watermark_models(
    img_dec_path: Optional[str],
    txt_dec_path: Optional[str],
    hash_file: Optional[str],
    device: torch.device,
) -> tuple:
    """Load watermark decoder models with optional integrity verification.
    Returns (None, None) if paths are not provided."""
    import json as _json

    hashes = {}
    if hash_file and os.path.exists(hash_file):
        with open(hash_file) as f:
            hashes = _json.load(f)

    wm_img_dec = None
    wm_txt_dec = None

    if img_dec_path and os.path.exists(img_dec_path):
        if 'wm_img_dec' in hashes:
            verify_model(img_dec_path, hashes['wm_img_dec'])
        wm_img_dec = ImageWatermarkDecoder().to(device)
        wm_img_dec.load_state_dict(torch.load(img_dec_path, map_location=device))
        wm_img_dec.eval()

    if txt_dec_path and os.path.exists(txt_dec_path):
        if 'wm_txt_dec' in hashes:
            verify_model(txt_dec_path, hashes['wm_txt_dec'])
        wm_txt_dec = TextWatermarkDecoder().to(device)
        wm_txt_dec.load_state_dict(torch.load(txt_dec_path, map_location=device))
        wm_txt_dec.eval()

    return wm_img_dec, wm_txt_dec


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)
        
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch

  

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    
    TP_all_multicls = np.zeros(4, dtype = int)
    TN_all_multicls = np.zeros(4, dtype = int)
    FP_all_multicls = np.zeros(4, dtype = int)
    FN_all_multicls = np.zeros(4, dtype = int)
    F1_multicls = np.zeros(4)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = logits_multicls[:, cls_idx]
            cls_pred[cls_pred>=0]=1
            cls_pred[cls_pred<0]=0
            
            TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
            TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
            FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
            FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()
            
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()
                 
    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)
    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)
    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
            
    for cls_idx in range(logits_multicls.shape[1]):
        Precision_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx])
        Recall_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx])
        F1_multicls[cls_idx] = 2*Precision_multicls*Recall_multicls / (Precision_multicls + Recall_multicls)            

    return AUC_cls, ACC_cls, EER_cls, \
        MAP.item(), OP, OR, OF1, CP, CR, CF1, F1_multicls, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #### Model #### 
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    if args.log:
        print(f"Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   

    checkpoint_dir = f'{args.output_dir}/{args.log_num}/checkpoint_{args.test_epoch}.pth'
    checkpoint = torch.load(checkpoint_dir, map_location='cpu') 
    state_dict = checkpoint['model']                       

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
                   
    # model.load_state_dict(state_dict)  
    if args.log:
        print('load checkpoint from %s'%checkpoint_dir)  
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)  

    # Load watermark decoders (optional — graceful fallback if not available)
    wm_img_dec, wm_txt_dec = load_watermark_models(
        img_dec_path=getattr(args, 'wm_img_dec', None),
        txt_dec_path=getattr(args, 'wm_txt_dec', None),
        hash_file=getattr(args, 'hash_file', None),
        device=device,
    )

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    
    if args.distributed:  
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                                samplers,
                                batch_size=[config['batch_size_val']], 
                                num_workers=[4], 
                                is_trains=[False], 
                                collate_fns=[None])[0]

    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start evaluation")

    AUC_cls, ACC_cls, EER_cls, \
    MAP, OP, OR, OF1, CP, CR, CF1, F1_multicls, \
    IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    ACC_tok, Precision_tok, Recall_tok, F1_tok  = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

    # Log watermark decoder availability
    if args.log:
        wm_available = wm_img_dec is not None and wm_txt_dec is not None
        logger.info(f"Watermark decoders available: {wm_available}")
        if wm_available:
            logger.info("Trust score = 0.7 * hammer_score + 0.3 * watermark_score")
        else:
            logger.info("Watermark decoders not loaded — trust score = hammer_score only")
    #============ evaluation info ============#
    val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                    "ACC_cls": "{:.4f}".format(ACC_cls*100),
                    "EER_cls": "{:.4f}".format(EER_cls*100),
                    "MAP": "{:.4f}".format(MAP*100),
                    "OP": "{:.4f}".format(OP*100),
                    "OR": "{:.4f}".format(OR*100),
                    "OF1": "{:.4f}".format(OF1*100),
                    "CP": "{:.4f}".format(CP*100),
                    "CR": "{:.4f}".format(CR*100),
                    "CF1": "{:.4f}".format(CF1*100),
                    "F1_FS": "{:.4f}".format(F1_multicls[0]*100),
                    "F1_FA": "{:.4f}".format(F1_multicls[1]*100),
                    "F1_TS": "{:.4f}".format(F1_multicls[2]*100),
                    "F1_TA": "{:.4f}".format(F1_multicls[3]*100),
                    "IOU_score": "{:.4f}".format(IOU_score*100),
                    "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                    "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                    "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                    "ACC_tok": "{:.4f}".format(ACC_tok*100),
                    "Precision_tok": "{:.4f}".format(Precision_tok*100),
                    "Recall_tok": "{:.4f}".format(Recall_tok*100),
                    "F1_tok": "{:.4f}".format(F1_tok*100),
    }
    
    if utils.is_main_process(): 
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': args.test_epoch,
                    }             
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)
    parser.add_argument('--wm_img_dec', default='', type=str, help='Path to image watermark decoder weights')
    parser.add_argument('--wm_txt_dec', default='', type=str, help='Path to text watermark decoder weights')
    parser.add_argument('--hash_file', default='model_hashes.json', type=str, help='Path to model integrity hash file')

    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    with open(args.config, 'r') as f:
        config = yaml.load(f)
 
    main_worker(0, args, config)
