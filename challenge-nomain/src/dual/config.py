import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='../data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='../data/test_b.json')
    parser.add_argument('--train_zip_feats', type=str, default='../data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='../data/test_b.zip')
    parser.add_argument('--test_output_csv', type=str, default='../data/result.csv')
    parser.add_argument('--val_ratio', default=0.00001, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=28, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=28, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=28, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    # parser.add_argument('--pretrain_model', type=str, default='save/model_epoch_3_loss_0.7648364305496216.bin')
    parser.add_argument('--savedmodel_path', type=str, default='save/saved_model')
    # parser.add_argument('--ckpt_file', type=str, default='save/albef/checkpoint/model_swa_0.670.bin')
    parser.add_argument('--model1_ckpt_file', type=str, default='save/saved_model/checkpoint/model.bin')
    parser.add_argument('--model2_ckpt_file', type=str, default='../simple/save/FGM/checkpoint/model.bin')
    parser.add_argument('--model3_ckpt_file', type=str, default='save/saved_model/model_epoch_2.bin')
    parser.add_argument('--model4_ckpt_file', type=str, default='../simple/save/FGM/model_epoch_1.bin')
    parser.add_argument('--model6_ckpt_file', type=str, default='../simple/save/PGD/checkpoint/model.bin')
    parser.add_argument('--model8_ckpt_file', type=str, default='../simple/save/PGD/model_epoch_1.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=4, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=700, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_hidden_size', type=int, default=768)
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=384)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=3840, help='nextvlad output size using dense') # swa670是768
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=768, help="linear size before final linear")
    parser.add_argument('--cat_size', type=int, default=3072, help='output size')

    return parser.parse_args()
