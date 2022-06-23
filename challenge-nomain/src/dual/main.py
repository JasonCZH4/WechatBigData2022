import logging
import os
import time
import copy
import torch
from config import parse_args
from data_helper import create_dataloaders
from FGM import PGD, EMA, FGM
from model import ALBEF
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

torch.cuda.set_device(0)


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_save_dir = os.listdir(base_dir)
    model_lists = []
    for _file in model_save_dir:
        if _file.endswith('.bin'):
            model_lists.append(os.path.join(base_dir, _file))

    # model_lists = sorted(model_lists,
    #                      key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists


def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

    assert 0 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            print(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu'))['model_state_dict'])
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    print(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.bin')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = ALBEF(args)
    swa_raw_model = copy.deepcopy(model)
    optimizer, scheduler = build_optimizer(args, model)
    # checkpoint = torch.load(args.pretrain_model, map_location='cpu')
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    ema = EMA(model, 0.999)
    ema.register()
    pgd = PGD(model)
    K = 3
    fgm = FGM(model)
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            fgm.attack()
            loss_sum, _, _, _ = model(batch)
            loss_sum.backward()
            fgm.restore()
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv, accuracy_adv, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(
                    f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        ema.apply_shadow()
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        if not os.path.exists(args.savedmodel_path):
            os.mkdir(args.savedmodel_path)
        mean_f1 = results['mean_f1']
        # if mean_f1 > best_score:
        # best_score = mean_f1
        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                   f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
        ema.restore()
    swa(swa_raw_model, args.savedmodel_path, swa_start=1)


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
