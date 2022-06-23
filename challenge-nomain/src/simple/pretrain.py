import logging
import os
import time
import torch
import torch.optim.swa_utils


from src.simple.config.config import parse_args
from src.simple.data_helpers.data_helper_pre import create_dataloaders
from models.pretrainmodel import QQModel4

from util import setup_device, setup_seed, setup_logging, build_optimizer


def train_and_validate(args):
    # 1. load data
    train_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = QQModel4(args, model_path=args.bert_dir)

    optimizer, scheduler = build_optimizer(args, model)

    # checkpoint = torch.load(args.pretain_load_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())

    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(0, args.pretrain_max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, masked_lm_loss, itm_loss = model(batch)

            loss = loss.mean()
            masked_lm_loss = masked_lm_loss.mean()
            itm_loss = itm_loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f} "
                             f"masked_lm_loss: {masked_lm_loss} itm_loss:{itm_loss}")

        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_dict': optimizer.state_dict()
                       , 'scheduler': scheduler.state_dict()},
                   f'{args.savepremodel_path}/model_epoch_{epoch}.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savepremodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
