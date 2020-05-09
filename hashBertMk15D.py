"""


 This program is designed to help fine tune  a pretrained distillbert model to performed Sem-eval 2016 Task 6 Sub-task A. The Program is based off of the run_glue.py script from Hugging Face provide at https://github.com/huggingface/transformers/tree/master/examples. 
 This 
--model_type: This specifies the type of model to be used, should be set as distilbert
--model_name_or_path: The pre-trained Hugging face bert variant to be used
	Has been tested with distil-bert-base-cased, other variant not guaranteed to work
--data_dir: should be a file path to a folder containg folder for traing data, as well as teh gold labedled evaluation data
--max_sequence_length: the maximum length sequence will be set too
--per gpu_train_batch_size: How large each processing batch is.
--learning_rate: The rate ate which the network adjusts it’s gradient.
--num_train_epoches: How many epochs the program will run over
--output_dir: use to specify the path of a folder where results and caches can be stored

In addition the following flags can be passed to the program
--do_traing: instructs the program to train on the input before evaluation
--do_eval : instructs the Program to run evaluation
--overwrite_cach instructs the program to overwrite cached data, such as the features lists stored in the output_dir.


"""
import argparse
import glob
import json
import logging
import os
import random
import pickle
import re
import csv
import math
import re


import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset,Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    InputExample
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import DataProcessor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

device=torch.device("cuda")
print(device)
DEBUG=1
logger = logging.getLogger(__name__)


#get the overal accracy of the sytem by taking the number of correct answer/total questions
# Sourced form Hugging faces glue_compute_metrics
def simple_accuracy(preds, labels):
        print("labels where:{}".format(labels))
        print("preds where:{}".format(preds))
        print("prediction match up was:{}".format(preds==labels))
        return (preds == labels).mean()

#sets up a seed for all the random number genrator, so that the results  of a run can be reproduced if the program is given the same seed

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

#trains Bert on the data set
#Modfied version of Hugging faces run_gule.py train() 

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    #in our case we will be loading up a pre-trained bert class
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)


    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

#takes writes the output of the network to a file
def write_preds(examples, preds):
    tag="notatag"
    #clears the files from the prediction directory
    # code sourced from https://stackoverflow.com/questions/1995373/deleting-all-files-in-a-directory-with-python/1995397
    filelist = [ f for f in os.listdir("preds")  ]
    for f in filelist:
        os.remove(os.path.join("preds", f))
    for i in range(0,len(preds)):
        splitID=re.split("%",examples[i].guid)
        #if were on a new tag we need to switch which file where writing to
        if(tag!=splitID[0]):
            if(i>0):
                tsvF.close()
            tag=splitID[0]
            
            filename="preds/"+tag+"_PREDICT.tsv"
            tsvF=open(filename,'a',newline='')
            tsv_writer = csv.writer(tsvF, delimiter='\t',quoting=csv.QUOTE_NONE, escapechar=None)
        tsv_writer.writerow([splitID[1],splitID[2],preds[i]])

#evaluates the netowork using the dev data
def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    
    results = {}
    
    #Load 
    eval_dataset,examples = load_and_cache_examples(args, tokenizer, evaluate=True)

    #find or output directory
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        #use our model to make yses no predeiction about the eval questions
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            #print("inputs ids:{} ,attention_mask:{}, labels:{}".format(batch[0],batch[1],batch[3]))
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        
    eval_loss = eval_loss / nb_eval_steps
    #use argmax to normalize the resulting prediction to eitehr 0 (no) or 1 (yes)
    preds = np.argmax(preds, axis=1)
    write_preds(examples,preds)
    oPred=[]
    oOut=[]
    #cuts predictions down to those in the Gold Labeled set
    #ie only tweet pairs that evalutated to 1 (first tweet is funnier)
    for i in range(0 , len(out_label_ids)):
        if(out_label_ids[i]==1):
            oPred.append(preds[i])
            oOut.append(out_label_ids[i])
    result = {"acc":simple_accuracy(np.asarray(oPred),np.asarray(oOut))}

    print("res where:{}".format(result))
    results.update(result)
    #save our results to the output directory
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def csv_to_list(filePath):
    #creates a list of every csv file in the directory
    csv_list= os.listdir(filePath)
    target_hashtags = [os.path.splitext(cFile)[0] for cFile in csv_list]
    #this list will hold lists corsiponding to the funniest tweet for each hash tag
    bestL=[]
    #this list will hold ilsts for tweets in the top 9 for each hastag
    topL=[]
    #this list will hold lists for tweets that are not in the top 10
    notL=[]

    #open up the csv file
    for tag in target_hashtags:
        tagBest=[]
        tagTop=[]
        tagNot=[]
        csv_file=open(filePath+"/"+tag+".tsv",newline="",encoding="utf8")
        csvLines=csv.reader(csv_file,delimiter="\t")
        
    #add every tweet from the file into a list of ditonarys
        for line in csvLines:
            
            score=line[2]
            entry= {"id":line[0],"tweet":line[1],"score":score,"tag":tag}
            if(score=="2"):
                tagBest.append(entry)
            elif(score=="1"):
                tagTop.append(entry)
            elif(score=="0"):
                tagNot.append(entry)
            else:
                print("error no score when adding tweet\n")
        csv_file.close()
        bestL.append(tagBest)
        topL.append(tagTop)
        notL.append(tagNot)

    """
    if(DEBUG):
        print("bestL:{} topl:{} notl:{}\n".format(len(bestL),len(topL),len(notL)))
    """
    return bestL ,topL ,notL

        
#degub script borrowed from the offical eval script
def nCr(n, r):
    '''http://stackoverflow.com/a/4941846'''
    f = math.factorial
    return f(n) / f(r) / f(n - r)


#This class is responsible for loading data from the ttraing, dev and test files for use by Bert
# Modeled after the DataProssecor class from hugging face
#url:https://github.com/huggingface/transformers/blob/a21d4fa410dc3b4c62f93aa0e6bbe4b75a101ee9/examples/hans/utils_hans.py#L89
#
class BoolQProcessor(DataProcessor):
    
    #Retrives all the traing examples as a list of dictonarys
    def get_train_examples(self, data_dir):
        trainPath=data_dir+"/train_dir/train_data"
        #trainPath=data_dir+"/train_dir/train_lite"
        
        #loads all the examples from the traing file into a list
        bestL,topL,notL=csv_to_list(trainPath)
        return self._create_examples(bestL,topL,notL, "train")

    #Retrives all the dev examples as a list of dictonarys
    def get_dev_examples(self, data_dir):
        """See base class."""
        #devPath=data_dir+"/trial_dir/trial_data"
        devPath=data_dir+"/Gold"
        bestL,topL,notL=csv_to_list(devPath)
        return self._create_examples(bestL,topL,notL, "dev")

    #returns the labels for the twwet
    # 2: the funnest tweet for a hashtag
    #1: a twweet that was in the top 10 funniest for a hashtag
    #0: a tweet that did not make the top 10
    def get_labels(self):
        return ["0", "1"]

    #this function takes in two tweet dictonaries and creates a comparison example
    #tweetLa and tweetLb, shuold be list of tweets dictonreis
    # all tweeets in a given list should have the same score
    # each tweet list should have diffrent score
    # if tweetLa scores > than tweetLb then the score should be 1 othersise it should be zero
    def create_comp(self,examples,tweetLa,tweetLb,score):
        for tweeta in tweetLa:
            for tweetb in tweetLb:
                #genrate a unqie id for the combo by concatinging thier tweet ID
                guid=tweeta.get("tag")+"%"+tweeta.get("id")+"%"+tweetb.get("id")
                
                #finds the hastag in the tweet and repalaces it it with a version seprated by spaces.
                #in addition the hastag is sperated from the rest of the tweet with a starting <hashtag> token and an ending </hashtag> token
                hashTag="<hashtag> "+re.sub(r"_",r" ",tweeta.get("tag"))+" </hashtag>"

                normaA=re.sub(r"\#[\w]+",hashTag,tweeta.get("tweet"))
                
                normaB=re.sub(r"\#[\w]+",hashTag,tweetb.get("tweet"))
                examples.append(InputExample(guid=guid,text_a=normaA,text_b=normaB,label=score))
        return examples  
    

    #gose though each list of tweets and combines them to get every uiqe combination of tweets with diffrent score
    def _create_examples(self, bestL,topL,notL, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        #gose though each hastag and creates all possible unqie combinations of 
        for i in range(0,len(bestL)):
        
            best=bestL[i]
            top=topL[i]
            notT=notL[i]
            if (DEBUG):
                nb_all_tweets = len(best) + len(top) + len(notT)
                print("in bestL there are{} tweets ; in topL their are {} tweets; in notL there are {} tweets; in total there are {} tweets".format(len(best),len(top),len(notT),nb_all_tweets))
            examples=self.create_comp(examples,best,top,"1")
            
            examples=self.create_comp(examples,top,best,"0")
            
            examples=self.create_comp(examples,best,notT,"1")
            
            examples=self.create_comp(examples,notT,best,"0")
            
            examples=self.create_comp(examples,top,notT,"1")
            
            examples=self.create_comp(examples,notT,top,"0")
            if(DEBUG):
                print("There are now {} combinations!".format(len(examples)))
                print("on iteration {} of {}".format(i,len(bestL)))
        if(DEBUG):
            print("done genrating examples\n")
        return examples



#loads up the example questions into a tensor for use by bert
def load_and_cache_examples(args, tokenizer, evaluate):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    #sets up a processor that reads and handles data from the file
    processor = BoolQProcessor()

    #tells hugging face that we want to run BERT as a classifier
    output_mode = "classification"
    # Load data features from cache or dataset file

    #genrates a file path to store our feture cache
    filename="chacheFile"
    directory=args.output_dir
    cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + "_" + filename
        )
    #searches to see if theres already a file with features so we don't have to regenrate them
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("found Cache")
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    #otherwise we genrate them now
    else:
        if os.path.exists(directory)==False:
            os.mkdir(directory)
        f=open(cached_features_file,"a")
        f.close()
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        #loads the apropritate examples from thier fiels etheir traing or dev
        print("eve={}".format(evaluate))
        if(evaluate==True):

            examples = processor.get_dev_examples(args.data_dir) 
            #print("first example ID is:{}".format(examples[0]))
        else:
            examples=processor.get_train_examples(args.data_dir)
        if(DEBUG):
            print("converting exampls to feateres!\n")
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )
        if(DEBUG):
            print("converstion done\n")
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if(DEBUG):
        print("converting data set to tensors!\n")
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    #print("example before exit :{}".format(examples[0]))
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset,examples





#this main function takes in argments from the command line and then uses them to set up the fine-tuning process
# Modfied version of the main function from Hugging faces run_glue.py
def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )


    args = parser.parse_args()
    args.n_gpu=1
    args.device=device
    num_labels=2
    #configrwe the pre trained model
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    #loads are intial pre-trained model and tokenizer from hugging face
    #model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
    #tokenizer breaks input into tokens and is also responsible for mask tokens
    tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(device)

    # begins the training process
    if args.do_train:
        #Loads up the traing set fron the given file
        traindata , examples=load_and_cache_examples(args,tokenizer,False)
        #print("train data:{}".format(traindata))
    

        # trians the model, returns the loss and the total number of steps taken in the processs
        global_step, tr_loss = train(args, traindata, model, tokenizer)
        print("finale traing loss was:{}\n there where{} steps\n".format(tr_loss,global_step))
        #output report on our progress
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        #saves out model
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)
    
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #loads the toknizer create masks 
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        #Loads up check points and processes the data as chunck starting at each check point
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            #runs evaluation on data starting at current check point.
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    #reports the final accuracy for the model.       
    print("accuracy was:{}".format(results))
    return results
if __name__ == "__main__":
    main()