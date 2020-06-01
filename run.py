from model import Model
import os
import argparse

def run(args,dataset1,data_dir,mode,task):
    args['task_name'] = task
    args['output_mode'] = mode
    for directory in os.listdir(data_dir):
        if not directory.startswith('.'):
            args['data_dir'] = data_dir+"/"+directory
            args['output_dir'] = "output_50eps/"+directory
            args['cache_dir'] = "cache/"+directory
            print("Experiment with ",directory)
            obj = Model(args)
            dataset = obj.load_and_cache_examples(args,obj.tokenizer)
            model,__, _ = obj.train(args,dataset,obj.model,obj.tokenizer)
            result,____ = obj.evaluate(args,model,obj.tokenizer)
            with open(str(dataset1)+'_result_50eps.txt',"a") as fp:
                fp.writelines(args['data_dir']+"\n")
                fp.writelines(str(result)+"\n")
            del obj
            del model
            del result




args = {
    'data_dir': '',
    'model_type':  'bert',
    'model_name': 'bert-base-uncased',
    'task_name': 'binary',
    'output_dir': 'outputs/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 64,
    'eval_batch_size': 64,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 50,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 5,
    'max_grad_norm': 2.0,

    'logging_steps': 90,
    'evaluate_during_training': True,
    'save_steps': 500,
    'eval_all_checkpoints': False,

    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'notes': ''
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",dest='dataset',help="imdb or sst2 or sts-b",metavar='str')
    parser.add_argument("--f",dest='path',help="data dir",metavar='path')
    parser.add_argument("--out",dest='mode',help="binary or regression",metavar='str')
    parser.add_argument("--task",dest='task',help="binary or sts-b",metavar='str')
    result = parser.parse_args()

    run(args,result.dataset,result.path,result.mode,result.task)
