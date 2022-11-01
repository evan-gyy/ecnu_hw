from encoder.bert_encoder import BERTEncoder
from models.rifre_sentence import RIFRE_SEN
from framework.sentence_re import Sentence_RE
from configs import Config
from utils import count_params
import numpy as np
import torch
import random, argparse
torch.cuda.set_device(0)

def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--test', default=True, type=bool)
    parser.add_argument('--dataset', default='semeval', type=str)
    args = parser.parse_args()
    dataset = args.dataset
    config = Config()
    if config.seed is not None:
        print(config.seed)
        seed_torch(config.seed)

    if dataset == 'semeval':
        print('test--' + dataset)
        config.class_nums = config.semeval_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base)
        model = RIFRE_SEN(sentence_encoder, config)
        count_params(model)
        framework = Sentence_RE(model,
                                train_path=config.semeval_train,
                                val_path=config.semeval_val,
                                test_path=config.semeval_test,
                                rel2id=config.semeval_rel2id,
                                pretrain_path=config.bert_base,
                                ckpt=config.semeval_ckpt,
                                batch_size=config.batch_size,
                                max_epoch=config.epoch,
                                lr=config.lr,
                                num_workers=4)
        framework.load_state_dict(config.semeval_ckpt)
        print('test:')
        framework.eval_semeval(framework.test_loader, mode='test')
        framework.save_pred_label(config.semeval_test)
    else:
        print('unknown dataset')