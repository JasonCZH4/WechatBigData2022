import json
import random
import zipfile
from io import BytesIO
from functools import partial
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer

from category_id_map import category_id_to_lv2id


def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,  # ann_path = data/annotations/labeled or test_a.json
                 zip_feats: str,  # zip_feats = data/zip_feats/labeled or test_a.zip
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        self.fc = nn.Linear(150, 50)
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer, args.bert_dir = chinese-macbert-base
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)  # 32, 768
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape  # num_frames 最多为 32(即最多为max_frames), feat_dim = 768

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)  # 32, 768
        mask = np.ones((self.max_frame,), dtype=np.int32)  # 32,
        if num_frames <= self.max_frame:  # 取前32帧
            feat[:num_frames] = raw_feats  # feat截取到前num_frames的raw_feats
            mask[num_frames:] = 0  # mask将num_frames后的值全置0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask  # 返回的是一个num_frames(最大为32), 768的feat矩阵;最长为32的二值向量,前num_frames为1后为0

    def tokenize_text(self, text1: str, text2: str) -> tuple:
        # encoded_inputs是一个3类字典[input_ids]; [token_type_ids]; [attention_mask]; 长度都为50
        encoded_inputs = self.tokenizer(text1, text2, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        token_type_ids = torch.LongTensor(encoded_inputs['token_type_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, token_type_ids, mask

    def title_asr_token_cut(self, text: str) -> tuple:
        if len(text) > 64:
            text = text[:64]
        return text

    def ocr_token_cut(self, text: str) -> tuple:
        if len(text) > 128:
            text = text[:128]
        return text

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)

        # Step 2, load title tokens
        # 可以拼接文本:+ self.anns[idx]['asr] + ''.join(i['text'] for i in self.anns[idx]['ocr'])
        # title = self.title_asr_token_cut(self.anns[idx]['title'])
        title = self.anns[idx]['title']
        # asr = self.title_asr_token_cut(self.anns[idx]['asr'])
        asr = self.anns[idx]['asr']
        # ocr = self.ocr_token_cut(''.join(i['text'] for i in self.anns[idx]['ocr']))
        ocr = ''.join(i['text'] for i in self.anns[idx]['ocr'])
        title_input, token_type_ids, title_mask = self.tokenize_text(title + '[SEP]' + asr, ocr)
        # title_input, title_mask = self.tokenize_text(title + asr + ocr)
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,    # 32, 768
            frame_mask=frame_mask,
            title_input=title_input,    # 260,
            title_mask=title_mask,
            token_type_ids=token_type_ids
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
