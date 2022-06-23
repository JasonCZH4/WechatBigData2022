import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import ALBEF
from Tencentmodel3 import QQModel3


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model1 = ALBEF(args)
    model2 = QQModel3(args, args.bert_dir)
    model3 = ALBEF(args)
    model4 = QQModel3(args, args.bert_dir)
    model6 = QQModel3(args, args.bert_dir)
    model8 = QQModel3(args, args.bert_dir)
    '''
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    # print(checkpoint)
    model.load_state_dict(checkpoint)
    '''
    checkpoint1 = torch.load(args.model1_ckpt_file, map_location='cpu')
    # model1.load_state_dict(checkpoint1['model_state_dict'])
    model1.load_state_dict(checkpoint1)

    checkpoint2 = torch.load(args.model2_ckpt_file, map_location='cpu')
    model2.load_state_dict(checkpoint2)

    checkpoint3 = torch.load(args.model3_ckpt_file, map_location='cpu')
    model3.load_state_dict(checkpoint3['model_state_dict'])

    checkpoint4 = torch.load(args.model4_ckpt_file, map_location='cpu')
    model4.load_state_dict(checkpoint4['model_state_dict'])

    checkpoint6 = torch.load(args.model6_ckpt_file, map_location='cpu')
    model6.load_state_dict(checkpoint6)

    checkpoint8 = torch.load(args.model8_ckpt_file, map_location='cpu')
    model8.load_state_dict(checkpoint8['model_state_dict'])


    if torch.cuda.is_available():
        model1 = torch.nn.parallel.DataParallel(model1.cuda())
    model1.eval()

    if torch.cuda.is_available():
        model2 = torch.nn.parallel.DataParallel(model2.cuda())
    model2.eval()

    if torch.cuda.is_available():
        model3 = torch.nn.parallel.DataParallel(model3.cuda())
    model3.eval()

    if torch.cuda.is_available():
        model4 = torch.nn.parallel.DataParallel(model4.cuda())
    model4.eval()

    if torch.cuda.is_available():
        model6 = torch.nn.parallel.DataParallel(model6.cuda())
    model6.eval()

    if torch.cuda.is_available():
        model8 = torch.nn.parallel.DataParallel(model8.cuda())
    model8.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        print('***********Merging Start!**************')
        for batch in dataloader:
            # try:
            pred_label_id1 = model1(batch, merge=True, train=False, inference=False)
            pred_label_id2 = model2(batch, merge=True, train=False, inference=False)
            pred_label_id3 = model3(batch, merge=True, train=False, inference=False)
            pred_label_id4 = model4(batch, merge=True, train=False, inference=False)
            pred_label_id6 = model6(batch, merge=True, train=False, inference=False)
            pred_label_id8 = model8(batch, merge=True, train=False, inference=False)
            # merge_pred = torch.add(pred_label_id1, pred_label_id2, alpha=1)
            merge_pred_temp = torch.add(pred_label_id1, pred_label_id2, alpha=1)
            merge_pred_temp = torch.add(merge_pred_temp, pred_label_id3, alpha=1)
            merge_pred_temp = torch.add(merge_pred_temp, pred_label_id4, alpha=1)
            merge_pred_temp = torch.add(merge_pred_temp, pred_label_id6, alpha=1)
            merge_pred = torch.add(merge_pred_temp, pred_label_id8, alpha=1)
            merge_pred = torch.div(merge_pred, 6)
            pred_label_id = torch.argmax(merge_pred, dim=1)
            predictions.extend(pred_label_id.cpu().numpy())
            # except:
            #     print(batch)
        print('***********Merging Complete!**************')

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
