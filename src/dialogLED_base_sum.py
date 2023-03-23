import os

import torch
import argparse
import torch.nn as nn
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import pandas as pd
import evaluate


scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore'),
            {'model_type': 'microsoft/deberta-xlarge-mnli'},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
        'bleurt': (
            evaluate.load('bleurt', config_name='BLEURT-20'),
            {},
            ['scores'],
            ['bleurt']
        ),
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description="sentence classification model")
    parser.add_argument(
        "--train_data", type=str, default="../data/parsed_data/train.json", help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/parsed_data/test.json",
        help="test file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/parsed_data/dev.json",
        help="test file"
    )
    parser.add_argument(
        "--train_data2", type=str, default=None, help="train file"
    )
    parser.add_argument(
        "--train_data3", type=str, default=None, help="train file"
    )
    parser.add_argument(
        "--dev_data2", type=str, default=None, help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_inference", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-base"
    )
    parser.add_argument(
        "--model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--output_file_name", type=str, default="taskB_Teddysum_run.csv"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_src_len", type=int, default=256
    )
    parser.add_argument(
        "--max_tgt_len", type=int, default=256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args



class DialogLEDBasedGeneration(nn.Module):
    def __init__(self, args, len_tokenizer):
        super(DialogLEDBasedGeneration, self).__init__()

        config = AutoConfig.from_pretrained(
            args.base_model,
            cache_dir=None,
            revision="main",
            use_auth_token=False,
        )

        # for DialogLED
        config.num_beams = 6
        config.max_length = args.max_tgt_len
        config.length_penalty = 2.0
        config.no_repeat_ngram_size = 3


        self.dialogLED = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model,
            config=config
        )

        self.dialogLED.resize_token_embeddings(len_tokenizer)
        self.config = self.dialogLED.config

    def forward(self, input_ids, attention_mask, target_ids, target_mask):

        outputs = self.dialogLED(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids[:, :-1],
            decoder_attention_mask=target_mask[:, :-1]
        )
        return outputs

    def generate(self, input_ids, tokenizer, max_length=1024):
        generated = self.dialogLED.generate(input_ids, num_beams=6, no_repeat_ngram_size=3, length_penalty=2, max_length=max_length)

        response = tokenizer.decode(generated[0], skip_special_tokens=True)

        return response

def tokenize_and_align_labels(tokenizer, src, tgt, max_src_len, max_tgt_len):

    data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'target_ids': [],
        'target_mask': []
    }

    tokenized_src_data = tokenizer(src, padding='max_length', max_length=max_src_len, truncation=True)

    tokenized_tgt_data = tokenizer(tgt, padding='max_length', max_length=max_tgt_len, truncation=True)

    data_dict['input_ids'].append(tokenized_src_data['input_ids'])
    data_dict['attention_mask'].append(tokenized_src_data['attention_mask'])

    data_dict['target_ids'].append(tokenized_tgt_data['input_ids'])
    data_dict['target_mask'].append(tokenized_tgt_data['attention_mask'])

    return data_dict


def get_dataset(raw_data, tokenizer, max_src_len, max_tgt_len):

    input_ids_list = []
    attention_mask_list = []
    target_ids_list = []
    target_mask_list = []

    for idx in range(len(raw_data)):

        dialogue = raw_data.iloc[idx, 2].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')

        note = raw_data.iloc[idx, 3]

        tokenized_data = tokenize_and_align_labels(tokenizer, dialogue, note, max_src_len, max_tgt_len)
        input_ids_list.extend(tokenized_data['input_ids'])
        attention_mask_list.extend(tokenized_data['attention_mask'])
        target_ids_list.extend(tokenized_data['target_ids'])
        target_mask_list.extend(tokenized_data['target_mask'])

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list), torch.tensor(target_ids_list), torch.tensor(target_mask_list))


def train(args=None):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('model would be saved at ', args.model_path)

    print('loading train data')
    train_data = pd.read_csv(args.train_data)
    dev_data = pd.read_csv(args.dev_data)

    print('train date len: ', len(train_data))
    print('dev date len: ', len(dev_data))

    if args.train_data2 is not None:
        train_data = pd.concat([train_data, pd.read_csv(args.train_data2)])
        print('two train date len: ', len(train_data))
    if args.train_data3 is not None:
        train_data = pd.concat([train_data, pd.read_csv(args.train_data3)])
        print('three train date len: ', len(train_data))
    if args.dev_data2 is not None:
        dev_data = pd.concat([dev_data, pd.read_csv(args.dev_data2)])
        print('two dev date len: ', len(dev_data))

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_dataloader = DataLoader(get_dataset(train_data, tokenizer, args.max_src_len, args.max_tgt_len), shuffle=True, batch_size=args.batch_size)
    # dev_dataloader = DataLoader(get_dataset(dev_data, tokenizer, args.max_src_len, args.max_tgt_len), shuffle=True,
    #                             batch_size=args.batch_size)

    print('loading model')
    model = DialogLEDBasedGeneration(args, len(tokenizer))
    model = torch.nn.DataParallel(model)

    # model.load_state_dict(
    #     torch.load('/root/data/saved_models/dialogLED_finetuned_to_meeting_0317/saved_model_epoch_46.pt',
    #                map_location=device))
    model.to(device)

    print(model)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas = (0.9, 0.999),
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        epoch_step+=1
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_target_ids, b_target_mask = batch

            model.zero_grad()

            output = model(b_input_ids, b_input_mask, b_target_ids, b_target_mask).logits

            loss = loss_fn(output.view(-1, model.module.config.vocab_size), b_target_ids[:, 1:].contiguous().view(-1))

            loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        model_saved_path = args.model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(model.state_dict(), model_saved_path)



        if args.do_eval:
            model.eval()
            predict_data = []
            evaluation_data = []

            for idx in range(len(dev_data)):

                dialogue = dev_data.iloc[idx, 2].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
                note = dev_data.iloc[idx, 3]

                tokenized_src_data = tokenizer(dialogue, padding='max_length', max_length=args.max_src_len,
                                               truncation=True)
                input_ids = torch.tensor([tokenized_src_data['input_ids']]).to(device)

                output = model.module.generate(input_ids, tokenizer, max_length=args.max_tgt_len)

                evaluation_data.append(note)
                predict_data.append(output)

            all_scores = {}
            for name, (scorer, kwargs, keys, save_keys) in scorers.items():
                scores = scorer.compute(references=evaluation_data, predictions=predict_data, **kwargs)
                for score_key, save_key in zip(keys, save_keys):
                    all_scores[save_key] = scores[score_key]


            print('validation result: ')
            print('rouge1: ', sum(all_scores['rouge1'])/len(all_scores['rouge1']))
            print('rouge2: ', sum(all_scores['rouge2']) / len(all_scores['rouge2']))
            print('rougeL: ', sum(all_scores['rougeL']) / len(all_scores['rougeL']))
            print('rougeLsum: ', sum(all_scores['rougeLsum']) / len(all_scores['rougeLsum']))
            print('bertscore_precision: ', sum(all_scores['bertscore_precision']) / len(all_scores['bertscore_precision']))
            print('bertscore_recall: ', sum(all_scores['bertscore_recall']) / len(all_scores['bertscore_recall']))
            print('bertscore_f1: ', sum(all_scores['bertscore_f1']) / len(all_scores['bertscore_f1']))
            print('bleurt: ', sum(all_scores['bleurt']) / len(all_scores['bleurt']))


            # print(result)
            #
            # total_loss = 0
            # for step, batch in enumerate(dev_dataloader):
            #     batch = tuple(t.to(device) for t in batch)
            #     b_input_ids, b_input_mask, b_target_ids, b_target_mask = batch
            #
            #     output = model(b_input_ids, b_input_mask, None, None).logits
            #
            #     loss = loss_fn(output.view(-1, model.module.config.vocab_size), b_target_ids[:, 1:].contiguous().view(-1))
            #
            #     total_loss += loss.item()
            #
            # avg_dev_loss = total_loss / len(dev_dataloader)
            # print("Average validation loss: {}".format(avg_dev_loss))


    print("training is done")


def inference(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    test_data = pd.read_csv(args.test_data)

    model = DialogLEDBasedGeneration(args, len(tokenizer))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.to(device)
    model.eval()


    data_dict = {
        "TestID": [],
        "SystemOutput": []
    }

    # data_dict = {
    #     "encounter_id": [],
    #     "note": [],
    #     'dialogue': []
    # }

    for idx in range(len(test_data)):
        dialogue = test_data.iloc[idx, 2].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
        # note = test_data.iloc[idx, 3]
        encounter_id = test_data.iloc[idx, 1]

        tokenized_src_data = tokenizer(dialogue, padding='max_length', max_length=args.max_src_len, truncation=True)
        input_ids = torch.tensor([tokenized_src_data['input_ids']]).to(device)

        output = model.module.generate(input_ids, tokenizer, max_length=1024)

        data_dict['TestID'].append(encounter_id)
        data_dict['SystemOutput'].append(output)

        # data_dict['encounter_id'].append(encounter_id)
        # data_dict['note'].append(output)
        # data_dict['dialogue'].append(dialogue)

    result = pd.DataFrame(data_dict)

    # result.to_csv(args.output_dir + args.output_file_name, index=False, encoding="utf-8-sig")
    result.to_csv(args.output_dir + args.output_file_name, index=False, encoding="utf-8-sig")


if __name__ == '__main__':

    args = parse_args()

    if args.do_train:
        train(args)
    elif args.do_inference:
        inference(args)

