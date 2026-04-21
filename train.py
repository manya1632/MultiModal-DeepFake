import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from ruamel.yaml import YAML
yaml = YAML()
import numpy as np
import random
import time
import datetime
import json
import hashlib
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Subset
from PIL import Image as PILImage

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

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.HAMMER import HAMMER
from models.watermark_image_encoder import ImageWatermarkEncoder
from models.watermark_image_decoder import ImageWatermarkDecoder
from models.watermark_text_encoder import TextWatermarkEncoder
from models.watermark_text_decoder import TextWatermarkDecoder
from utils.metrics import compute_psnr, compute_nc

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
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def compute_watermark_bits(text_list: list, device: torch.device) -> torch.Tensor:
    """Compute 128-bit watermark vectors from SHA256(text) for a batch.
    Returns (B, 128) float32 tensor with values in {0.0, 1.0}."""
    bits_list = []
    for text in text_list:
        digest = hashlib.sha256(text.encode()).digest()[:16]  # 16 bytes = 128 bits
        bits = torch.tensor([int(b) for byte in digest for b in format(byte, '08b')], dtype=torch.float32)
        bits_list.append(bits)
    return torch.stack(bits_list).to(device)


def compute_image_watermark_bits(image_features: torch.Tensor) -> torch.Tensor:
    """Compute 128-bit watermark vectors from image features hash for a batch.
    Returns (B, 128) float32 tensor with values in {0.0, 1.0}."""
    bits_list = []
    for feat in image_features:
        feat_bytes = feat.detach().cpu().float().numpy().tobytes()[:16]
        if len(feat_bytes) < 16:
            feat_bytes = feat_bytes + b'\x00' * (16 - len(feat_bytes))
        digest = hashlib.sha256(feat_bytes).digest()[:16]
        bits = torch.tensor([int(b) for byte in digest for b in format(byte, '08b')], dtype=torch.float32)
        bits_list.append(bits)
    return torch.stack(bits_list).to(image_features.device)


def apply_noise_augmentation(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply Gaussian noise and JPEG compression simulation to watermarked images.
    Gaussian noise: sigma in [0.0, 0.05]. JPEG quality: [70, 95]."""
    import random as rnd
    from torchvision import transforms

    # Gaussian noise
    sigma = rnd.uniform(0.0, 0.05)
    images = images + sigma * torch.randn_like(images)
    images = torch.clamp(images, 0.0, 1.0)

    # JPEG compression simulation (applied per-image in batch)
    quality = rnd.randint(70, 95)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    augmented = []
    for img in images.cpu():
        pil_img = to_pil(img.clamp(0, 1))
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        compressed = to_tensor(PILImage.open(buf))
        augmented.append(compressed)
    return torch.stack(augmented).to(device)


def freeze_encoders(model: torch.nn.Module) -> None:
    """Freeze visual_encoder and text_encoder weights in HAMMER model.
    Only fusion layers, classification head, and watermark modules remain trainable.
    Validates: Requirements 1.4, 1.5"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'visual_encoder' in name or 'text_encoder' in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} parameters in visual_encoder and text_encoder.")


def create_balanced_subset(dataset, subset_size: int):
    """Create a balanced subset with equal real and fake samples.
    Returns a torch.utils.data.Subset of size min(subset_size, 2*min(n_real, n_fake)).
    Validates: Requirements 1.1"""
    real_indices = []
    fake_indices = []

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            label = sample[1]  # label is second element
            if label == 'orig':
                real_indices.append(idx)
            else:
                fake_indices.append(idx)
        except Exception:
            continue

    n_per_class = min(len(real_indices), len(fake_indices), subset_size // 2)

    rng = np.random.default_rng(42)
    selected_real = rng.choice(real_indices, size=n_per_class, replace=False).tolist()
    selected_fake = rng.choice(fake_indices, size=n_per_class, replace=False).tolist()

    selected_indices = selected_real + selected_fake
    rng.shuffle(selected_indices)

    print(f"Balanced subset: {n_per_class} real + {n_per_class} fake = {len(selected_indices)} total")
    return Subset(dataset, selected_indices)


def train(args, model, wm_img_enc, wm_img_dec, wm_txt_enc, wm_txt_dec, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, summary_writer, scaler=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_MAC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_TMG', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_watermark', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('psnr', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('nc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    # FIX 5: lower print_freq so output appears frequently on Kaggle
    print_freq = 10
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    global_step = epoch * len(data_loader)

    # FIX 4: removed distributed sampler set_epoch — not needed for single-GPU

    print(f"🔥 Entering training loop... (epoch {epoch}, {len(data_loader)} batches)")

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)        

        optimizer.zero_grad()
  
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
 
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        use_fp16 = config.get('use_fp16', False) and scaler is not None

        with autocast(enabled=use_fp16):
            # --- Watermark step ---
            m_T = compute_watermark_bits(text, device)  # (B, 128) from SHA256(text)

            # Get image features for image watermark bits (use raw image as proxy)
            m_I = compute_image_watermark_bits(image.mean(dim=[2, 3]))  # (B, 128)

            # Embed watermarks
            I_w = wm_img_enc(image, m_T)                    # (B, 3, 224, 224)

            # Get text embeddings from BERT for text watermarking
            with torch.no_grad():
                # Single-GPU: always use model directly (no .module wrapper)
                text_embeds = model.text_encoder.embeddings(
                    input_ids=text_input.input_ids
                )
            E_w = wm_txt_enc(text_embeds, m_I)              # (B, seq_len, 768)

            # Apply noise augmentation to watermarked images
            I_w_aug = apply_noise_augmentation(I_w.detach(), device)

            # Decode watermarks
            m_T_hat = wm_img_dec(I_w_aug)                   # (B, 128)
            m_I_hat = wm_txt_dec(E_w)                       # (B, 128)

            # Watermark losses
            mse_loss = torch.nn.functional.mse_loss(I_w, image)
            bce_img = torch.nn.functional.binary_cross_entropy(m_T_hat, m_T)
            L_image = mse_loss + bce_img

            cos_sim = torch.nn.functional.cosine_similarity(
                text_embeds.view(text_embeds.size(0), -1),
                E_w.view(E_w.size(0), -1), dim=1
            ).mean()
            bce_txt = torch.nn.functional.binary_cross_entropy(m_I_hat, m_I)
            L_text = (1.0 - cos_sim) + bce_txt

            L_watermark = config.get('loss_watermark_wgt', 1.0) * (L_image + L_text)

            # --- HAMMER classification step (with watermarked inputs) ---
            loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = model(I_w, label, text_input, fake_image_box, fake_token_pos, alpha=alpha)

            L_classification = (config['loss_MAC_wgt'] * loss_MAC
                              + config['loss_BIC_wgt'] * loss_BIC
                              + config['loss_bbox_wgt'] * loss_bbox
                              + config['loss_giou_wgt'] * loss_giou
                              + config['loss_TMG_wgt'] * loss_TMG
                              + config['loss_MLC_wgt'] * loss_MLC)

            loss = L_watermark + L_classification

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler._scale < 1.0:
                print(f"[FP16] GradScaler skipped step at epoch {epoch}, iter {i} due to overflow.")
        else:
            loss.backward()
            optimizer.step()

        # Compute PSNR and NC metrics (detached, no grad)
        with torch.no_grad():
            psnr_val = compute_psnr(image.float(), I_w.float())
            nc_val = compute_nc(m_T.float(), (m_T_hat > 0.5).float())

        # FIX 6: debug print every 10 iters so Kaggle shows progress
        if i % 10 == 0:
            print(f"  Epoch {epoch}, Iter {i}/{len(data_loader)}, "
                  f"Loss {loss.item():.4f}, WM {L_watermark.item():.4f}, "
                  f"PSNR {psnr_val if psnr_val != float('inf') else 100.0:.1f}, NC {nc_val:.3f}",
                  flush=True)
        
        metric_logger.update(loss_MAC=loss_MAC.item())
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_TMG=loss_TMG.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_watermark=L_watermark.item())
        metric_logger.update(psnr=psnr_val if psnr_val != float('inf') else 100.0)
        metric_logger.update(nc=nc_val)
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step': 
            scheduler.step(i//step_size)   

        global_step+=1
        

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],                                                                                                  
                'loss_MAC': loss_MAC.item(),                                                                                                  
                'loss_BIC': loss_BIC.item(),                                                                                                  
                'loss_bbox': loss_bbox.item(),                                                                                                  
                'loss_giou': loss_giou.item(),                                                                                                  
                'loss_TMG': loss_TMG.item(),                                                                                                  
                'loss_MLC': loss_MLC.item(),                                                                                                  
                'loss': loss.item(),
                'loss_watermark': L_watermark.item(),
                'psnr': psnr_val if psnr_val != float('inf') else 100.0,
                'nc': nc_val,
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()   
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   
    
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

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
    
    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)
    
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

    return AUC_cls, ACC_cls, EER_cls, \
           MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
           IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
           ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    # FIX 3: removed init_dist — not needed for single-GPU Kaggle training

    log_dir = os.path.join(args.output_dir, 'log' + args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w'))

    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    # FIX 1: device setup inside main_worker with proper indentation
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    print("🔥 Starting training setup...")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']  
    best = 0
    best_epoch = 0  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)

    if config.get('subset_size'):
        if args.log:
            print(f"Creating balanced subset of size {config['subset_size']}")
        train_dataset = create_balanced_subset(train_dataset, config['subset_size'])
    
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                samplers,
                                batch_size=[config['batch_size_train']]+[config['batch_size_val']], 
                                num_workers=[4, 4], 
                                is_trains=[True, False], 
                                collate_fns=[None, None])

    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

    #### Model #### 
    if args.log:
        print(f"Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)

    # --- Watermark modules ---
    wm_img_enc = ImageWatermarkEncoder(watermark_dim=config.get('watermark_dim', 128),
                                        alpha=config.get('watermark_alpha', 0.03)).to(device)
    wm_img_dec = ImageWatermarkDecoder().to(device)
    wm_txt_enc = TextWatermarkEncoder(hidden_dim=768,
                                       watermark_dim=config.get('watermark_dim', 128),
                                       alpha=config.get('watermark_alpha', 0.03)).to(device)
    wm_txt_dec = TextWatermarkDecoder(hidden_dim=768,
                                       watermark_dim=config.get('watermark_dim', 128)).to(device)

    # Freeze encoders if configured
    if config.get('freeze_encoders', False):
        freeze_encoders(model)

    # FP16 scaler
    scaler = GradScaler() if config.get('use_fp16', False) else None

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    # Add watermark module parameters to optimizer
    wm_params = (list(wm_img_enc.parameters()) + list(wm_img_dec.parameters()) +
                 list(wm_txt_enc.parameters()) + list(wm_txt_dec.parameters()))
    optimizer.add_param_group({'params': wm_params, 'lr': config['optimizer']['lr']})
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
        # model.load_state_dict(state_dict)  
        if args.log:
            print('load checkpoint from %s'%args.checkpoint)  
        msg = model.load_state_dict(state_dict, strict=False)
        if args.log:
            print(msg)  

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
            
        train_stats = train(args, model, wm_img_enc, wm_img_dec, wm_txt_enc, wm_txt_dec, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, summary_writer, scaler=scaler) 
        AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok \
        = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
                'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
                'EER_cls': round(EER_cls*100, 4),                                                                                                  
                'MAP': round(MAP*100, 4),                                                                                                  
                'OP': round(OP*100, 4),                                                                                                  
                'OR': round(OR*100, 4), 
                'OF1': round(OF1*100, 4), 
                'CP': round(CP*100, 4), 
                'CR': round(CR*100, 4), 
                'CF1': round(CF1*100, 4), 
                'OP_k': round(OP_k*100, 4), 
                'OR_k': round(OR_k*100, 4), 
                'OF1_k': round(OF1_k*100, 4), 
                'CP_k': round(CP_k*100, 4), 
                'CR_k': round(CR_k*100, 4), 
                'CF1_k': round(CF1_k*100, 4), 
                'IOU_score': round(IOU_score*100, 4),                                                                                                  
                'IOU_ACC_50': round(IOU_ACC_50*100, 4),                                                                                                  
                'IOU_ACC_75': round(IOU_ACC_75*100, 4),                                                                                                  
                'IOU_ACC_95': round(IOU_ACC_95*100, 4),                                                                                                  
                'ACC_tok': round(ACC_tok*100, 4),                                                                                                  
                'Precision_tok': round(Precision_tok*100, 4),                                                                                                  
                'Recall_tok': round(Recall_tok*100, 4),                                                                                                  
                'F1_tok': round(F1_tok*100, 4),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

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
                     "OP_k": "{:.4f}".format(OP_k*100),
                     "OR_k": "{:.4f}".format(OR_k*100),
                     "OF1_k": "{:.4f}".format(OF1_k*100),
                     "CP_k": "{:.4f}".format(CP_k*100),
                     "CR_k": "{:.4f}".format(CR_k*100),
                     "CF1_k": "{:.4f}".format(CF1_k*100),
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
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                        }             
            with open(os.path.join(log_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'config': config,
                    'epoch': epoch,
                }                    
            if (epoch % args.model_save_epoch == 0 and epoch!=0):
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch)) 
            if float(val_stats['AUC_cls'])>best:
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_best.pth')) 
                best = float(val_stats['AUC_cls'])
                best_epoch = epoch 

        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch+warmup_steps+1)
        # FIX 8: removed dist.barrier() — hangs on single-GPU, not needed

    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch))   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str)) 
        with open(os.path.join(log_dir, "log.txt"),"a") as f:
            f.write("best epoch: {}, Training time: {}".format(best_epoch, total_time_str))    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--log', default=False, action='store_true')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # main(args, config)
    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))