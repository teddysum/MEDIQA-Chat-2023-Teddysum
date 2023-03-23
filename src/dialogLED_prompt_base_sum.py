import json
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
import re
from collections import OrderedDict
import pprint as pp
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

cmd_token = '<COMMAND>'

special_tokens_dict = {
    'additional_special_tokens': [cmd_token]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)


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

subjective_subsections = OrderedDict()
subjective_subsections[ 'REASON FOR VISIT :' ] = [ 'cc :', 'chief complaint :', 'reason for visit :' ]
subjective_subsections[ 'HISTORY OF PRESENT ILLNESS :' ] = [ 'history :', 'history of present illness :', 'history of present illness', 'hpi :', 'hpi', 'hpi notes :', 'interval history :', 'interval hx :', 'subjective :' ]
subjective_subsections[ 'REVIEW OF SYSTEM :' ] = [ 'ros :', 'review of system :', 'review of systems :' ]

objectiveexam_subsections = OrderedDict()
objectiveexam_subsections[ 'PHYSICAL EXAMINATION :' ] = [ 'physical exam :', 'physical examination :', 'pe :', 'physical findings :', 'examination :', 'exam :' ]

objectiveresults_subsections = OrderedDict()
objectiveresults_subsections[ 'FINDINGS :' ] = [ 'results :', 'findings :' ]

ap_subsections = OrderedDict()
ap_subsections[ 'ASSESSMENT :' ] = ['assessment :', 'a:' ]
ap_subsections[ 'PLAN OF CARE :' ] = [ 'plan :', 'plan of care :', 'p:', 'medical decision-making plan :', 'summary plan' ]
ap_subsections[ 'ASSESSMENT AND PLAN :' ] = [ 'ap :', 'a / p :', 'assessment and plan :', 'assessment & plan :', 'disposition / plan :' ]

sectcat2subsections = OrderedDict()
sectcat2subsections[ 'subjective' ] = subjective_subsections
sectcat2subsections[ 'objective_exam' ] = objectiveexam_subsections
sectcat2subsections[ 'objective_results' ] = objectiveresults_subsections
sectcat2subsections[ 'assessment_and_plan' ] = ap_subsections

NOSECTIONHEADER = 'default'
subsectionheader2section = {}
for sh, sshdicts in sectcat2subsections.items() :
    for ssh, lst in sshdicts.items():
        subsectionheader2section[ssh] = sh
subsectionheader2section[ NOSECTIONHEADER ] = NOSECTIONHEADER


section_dict = {
 'ASSESSMENT': 'ASSESSMENT AND PLAN',
 'ASSESSMENT AND PLAN': 'ASSESSMENT AND PLAN',
 'PLAN': 'ASSESSMENT AND PLAN',

 'EXAM': 'PHYSICAL EXAMINATION',
 'PHYSICAL EXAM': 'PHYSICAL EXAMINATION',
 'PHYSICAL EXAMINATION': 'PHYSICAL EXAMINATION',

 'HISTORY OF PRESENT ILLNESS': 'HISTORY OF PRESENT ILLNESS',
 'HPI:': 'HISTORY OF PRESENT ILLNESS',

 'REVIEW OF SYSTEMS': 'REVIEW OF SYSTEMS',

 'CC:': 'CHIEF COMPLAINT',
 'CHIEF COMPLAINT': 'CHIEF COMPLAINT',

 'RESULTS': 'RESULTS',

 'CURRENT MEDICATIONS': 'MEDICATIONS',
 'CURRENT MEDICATIONS:': 'MEDICATIONS',
 'MEDICATIONS': 'MEDICATIONS',

 'PAST HISTORY': 'HISTORY',
 'PAST MEDICAL HISTORY:': 'HISTORY',
 'MEDICAL HISTORY': 'HISTORY',
 'SOCIAL HISTORY': 'HISTORY',
 'SURGICAL HISTORY': 'HISTORY',
 'FAMILY HISTORY': 'HISTORY',

 'IMPRESSION': 'IMPRESSION',

 'INSTRUCTIONS': 'INSTRUCTIONS',

 'VITALS': 'VITALS',
 'VITALS REVIEWED': 'VITALS'
}

section_list = [
    'ASSESSMENT AND PLAN', 'PHYSICAL EXAMINATION', 'HISTORY OF PRESENT ILLNESS', 'REVIEW OF SYSTEMS', 'CHIEF COMPLAINT', 'RESULTS', 'MEDICATIONS', 'HISTORY', 'IMPRESSION', 'INSTRUCTIONS', 'VITALS'
]

def compile_regexexpression(vlst):
    expressions = []
    otherexps = []
    for exp in vlst:
        exp2 = '(' + re.escape(exp).replace('\ ', '\s*') + ')'
        expressions.append(exp2)
        if exp[-1] == ':':
            # allow without : if the line is empty
            exp2 = '(' + re.escape(exp[:-1]).replace('\ ', '\s*') + ')'
            otherexps.append(exp2)

    patt = '\s*(?P<sectionheader1>' + '|'.join(expressions) + ').*'
    if len(otherexps) > 0:
        pattott = '\s*(?P<sectionheader2>' + '|'.join(otherexps) + ')\s*$'
        return '(' + patt + '|' + pattott + ')'

    return patt

subsect2regex = {}
for _, sectcat2subsections in sectcat2subsections.items():
    for subsect, vlst in sectcat2subsections.items():
        subsect2regex[subsect] = compile_regexexpression(vlst)


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
        config.max_length = 360  # AMI: 360; ICSI: 512; QMSum: 128; FD: 256; TMS: 360
        config.length_penalty = 2.0
        config.no_repeat_ngram_size = 3


        self.dialogLED = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model,
            config=config
        )
        # self.dialogLED = AutoModelForSeq2SeqLM.from_pretrained('MingZhong/DialogLED-large-5120')

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

    section_summary_dict = {

    }

    for section in section_list:
        section_summary_dict[section] = ""

    for line in tgt.split('\n'):
        if line in section_dict:
            current_section = section_dict[line]
            if current_section == 'HISTORY':
                section_summary_dict[current_section] = section_summary_dict[current_section] + line + '\n'
            continue
        if len(line) > 1:
            section_summary_dict[current_section] = section_summary_dict[current_section] + line + '\n'

    for section, contents in section_summary_dict.items():

        input = src + cmd_token + "Based on this conversation, make a summary of the " + section

        if contents == '':
            contents == "None"
            output = ""
        else:
            output = section + '\n\n' +contents

        tokenized_src_data = tokenizer(input, padding='max_length', max_length=max_src_len, truncation=True)
        tokenized_tgt_data = tokenizer(output, padding='max_length', max_length=max_tgt_len, truncation=True)

        data_dict['input_ids'].append(tokenized_src_data['input_ids'])
        data_dict['attention_mask'].append(tokenized_src_data['attention_mask'])

        data_dict['target_ids'].append(tokenized_tgt_data['input_ids'])
        data_dict['target_mask'].append(tokenized_tgt_data['attention_mask'])

    all_section = ', '.join(section_list)
    input = src + cmd_token + "Based on this conversation, make a summary of " + all_section

    tokenized_src_data = tokenizer(input, padding='max_length', max_length=max_src_len, truncation=True)
    tokenized_tgt_data = tokenizer(tgt, padding='max_length', max_length=max_tgt_len, truncation=True)

    data_dict['input_ids'].append(tokenized_src_data['input_ids'])
    data_dict['attention_mask'].append(tokenized_src_data['attention_mask'])

    data_dict['target_ids'].append(tokenized_tgt_data['input_ids'])
    data_dict['target_mask'].append(tokenized_tgt_data['attention_mask'])

    return data_dict

def section_summary_dict_to_summary(section_summary_dict):

    summary = ""
    for section, contents in section_summary_dict.items():

        if contents in ["", "None", "none"]:
            continue
        if section == 'HISTORY':
            summary = summary + contents + '\n'
        else:
            summary = summary + section + '\n\n'
            summary = summary + contents + '\n'

    return summary


def note_preprocessing(note):
    text_list = []

    for linenum, line in enumerate(note.split('\n')):
        is_appended = False
        for main_header, header in subsect2regex.items():
            # print(main_header)
            # print(header)
            m = re.match(header, line, re.IGNORECASE)
            if m:
                line = line[:m.span()[0]] + main_header + line[m.span()[1]:]
                text_list.append(line)
                is_appended = True

        m = re.match('\s*IMPRESSION', line)
        if m:
            text_list.append('impression')
            is_appended = True

        if is_appended is False:
            text_list.append(line)

    text = '\n'.join(text_list)

    return text

# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


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

    print('train_classifier')
    print('model would be saved at ', args.model_path)

    print('loading train data')

    train_data = pd.read_csv(args.train_data)
    dev_data = pd.read_csv(args.dev_data)

    print('train date len: ', len(train_data))
    print('dev date len: ', len(dev_data))

    if args.train_data2 is not None:
        train_data = pd.concat([train_data, pd.read_csv(args.train_data2)])
        print('two train date len: ', len(train_data))
    if args.dev_data2 is not None:
        dev_data = pd.concat([dev_data, pd.read_csv(args.dev_data2)])
        print('two dev date len: ', len(dev_data))

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    train_dataloader = DataLoader(get_dataset(train_data, tokenizer, args.max_src_len, args.max_tgt_len), shuffle=True, batch_size=args.batch_size)
    # dev_dataloader = DataLoader(get_dataset(dev_data, tokenizer, args.max_src_len, args.max_tgt_len), shuffle=True,
    #                              batch_size=args.batch_size)

    print('loading model')
    model = DialogLEDBasedGeneration(args, len(tokenizer))
    model = torch.nn.DataParallel(model)

    # model.load_state_dict(
    #     torch.load('/root/data/saved_models/dialogLED_finetuned_to_meeting_0315/saved_model_epoch_4.pt',
    #                map_location=device))
    #
    model.to(device)
    # metric = load_metric("accuracy")

    print(model)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
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

    # label_smoother = LabelSmoother(epsilon=0.1)

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

                # section_summary_dict = {
                #
                # }
                #
                # for section in section_list:
                #     section_summary_dict[section] = ""
                #
                # for section in section_list:
                #
                #     input = dialogue + cmd_token + "Based on this conversation, make a summary of the " + section
                #
                #     tokenized_src_data = tokenizer(input, padding='max_length', max_length=args.max_src_len,
                #                                    truncation=True)
                #     input_ids = torch.tensor([tokenized_src_data['input_ids']]).to(device)
                #
                #     output = model.module.generate(input_ids, tokenizer, max_length=args.max_tgt_len)
                #
                #     section_summary_dict[section] = output
                #
                # summary = section_summary_dict_to_summary(section_summary_dict)

                all_section = ', '.join(section_list)
                input = dialogue + cmd_token + "Based on this conversation, make a summary of " + all_section
                tokenized_src_data = tokenizer(input, padding='max_length', max_length=args.max_src_len, truncation=True)
                input_ids = torch.tensor([tokenized_src_data['input_ids']]).to(device)

                output = model.module.generate(input_ids, tokenizer, max_length=args.max_tgt_len)

                evaluation_data.append(note)
                predict_data.append(output)

            # result = evaluation(predict_data, evaluation_data, evaluation_metrics=['ROUGE-1', 'BLEU'], ratio=1, iteration=1)
            # predict_data, evaluation_data = postprocess_text(predict_data, evaluation_data)
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
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = pd.read_csv(args.test_data)
    print(len(tokenizer))

    model = DialogLEDBasedGeneration(args, len(tokenizer))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.to(device)
    model.eval()


    data_dict = {
        "TestID": [],
        "SystemOutput": []
    }
    #
    # data_dict = {
    #     "encounter_id": [],
    #     "note": [],
    #     'dialogue': []
    # }

    for idx in range(len(test_data)):
        dialogue = test_data.iloc[idx, 2].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
        # note = test_data.iloc[idx, 3]
        encounter_id = test_data.iloc[idx, 1]

        all_section = ', '.join(section_list)
        input = dialogue + cmd_token + "Based on this conversation, make a summary of " + all_section
        tokenized_src_data = tokenizer(input, padding='max_length', max_length=args.max_src_len, truncation=True)
        input_ids = torch.tensor([tokenized_src_data['input_ids']]).to(device)

        output = model.module.generate(input_ids, tokenizer, max_length=args.max_tgt_len)


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

