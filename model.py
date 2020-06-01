from __future__ import absolute_import, division, print_function
import glob
import logging
import os
import random
import json
import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange,tqdm
from tensorboardX import SummaryWriter
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils import (convert_examples_to_features,output_modes, processors)
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix,f1_score
from scipy.stats import pearsonr, spearmanr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL CLASS
MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class Model:

    def __init__(self, args):

        # checking output directory
        if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))
        # Loading model and config
        if args['output_mode'] == "classification":
            num_labels = 2
        elif args['output_mode'] == "regression":
            num_labels = 1
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
        config = config_class.from_pretrained(args['model_name'], num_labels=num_labels, finetuning_task=args['task_name'])
        self.tokenizer = tokenizer_class.from_pretrained(args['model_name'])
        self.model = model_class.from_pretrained(args['model_name'],config=config)
        self.model.to(device)



    def load_and_cache_examples(self,args, tokenizer, evaluate=False):
        task = args['task_name']
        if task in processors.keys() and task in output_modes.keys():
            processor = processors[task]()
            label_list = processor.get_labels()
        else:
            raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')
        output_mode = args['output_mode']

        mode = 'dev' if evaluate else 'train'
        cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

        if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else:
            logger.info("Creating features from dataset file at %s", args['data_dir'])
            label_list = processor.get_labels()
            examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
            features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
    def train(self,args,train_dataset, model, tokenizer):
        tb_writer = SummaryWriter()
        if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
        
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        warmup_steps = math.ceil(t_total * args['warmup_ratio'])
        args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)

        if args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
            
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args['num_train_epochs'])
        logger.info("  Total train batch size  = %d", args['train_batch_size'])
        logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
        tr_loss_file = open(args['output_dir']+'/tr_loss.txt','w') 
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                print("\r%f" % loss, end='')
                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']
                    tr_loss_file.writelines(str(loss.item())+'\n')

                if args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                    
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                        # Log metrics
                        if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                            # training eval
                            results, _ = self.evaluate(args,model, tokenizer,evaluate=False,prefix="train")
                            for key, value in results.items():
                                tb_writer.add_scalar('train_{}'.format(key), value, global_step)
                            # val eval
                            results, _ = self.evaluate(args,model, tokenizer,evaluate=True,prefix="val")
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                        logging_loss = tr_loss

                    if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        #model_to_save.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


        return model, global_step, tr_loss / global_step
    def evaluate(self,args,model, tokenizer, evaluate = True, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args['output_dir']

        results = {}
        EVAL_TASK = args['task_name']

        eval_dataset = self.load_and_cache_examples(args,tokenizer, evaluate)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)


        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':     batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif args['output_mode'] == "regression":
            preds = np.squeeze(preds)
        result, wrong = self.compute_metrics(args,EVAL_TASK, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix+"_result.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** "+prefix+" results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return results, wrong
    def get_mismatched(self,labels, preds,args):
        processor = processors[args['task_name']]()
        mismatched = labels != preds
        examples = processor.get_dev_examples(args['data_dir'])
        wrong = [i for (i, v) in zip(examples, mismatched) if v]
        return wrong

    def get_eval_report(self,labels, preds,args):
        mcc = matthews_corrcoef(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        return {
            "mcc": mcc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }, self.get_mismatched(labels, preds,args)
    def pearson_and_spearman(self,preds, labels,args):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        },self.get_mismatched(labels,preds,args)

    def compute_metrics(self,args,task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name=="binary":
            return self.get_eval_report(labels, preds,args)
        elif task_name == "sts-b":
            return self.pearson_and_spearman(preds, labels,args)