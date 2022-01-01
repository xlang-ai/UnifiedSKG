# UnifiedSKG:books:: Unifying Structured Knowledge Grounding with Text-to-Text Language Models

![](https://img.shields.io/github/last-commit/HKUNLP/UnifiedSKG?color=green) ![](https://img.shields.io/badge/PRs-Welcome-red) 



<img src="pics/logos.png" align="middle" width="98%">

**S**tructured **k**nowledge **g**rounding (**SKG**) leverages structured knowledge to complete user requests, such as semantic parsing over databases and question answering over knowledge bases. Since the inputs and outputs of SKG tasks are heterogeneous, they were historically studied in separate by different communities,  which limits systematic and compatible research on SKG. In this paper, we overcome this limitation by proposing the **UNIFIEDSKG framework**, which unifies **21 SKG tasks** into the text-to-text format, aiming to promote systematic SKG research, instead of being exclusive to a single task, domain, or dataset. We show that large language models like T5, with simple modification when necessary, achieve **state-of-the-art performance on all 21 tasks**. **UNIFIEDSKG** facilitates the investigation of **multi-task, zero-shot, and few-shot learning**. We demonstrate that multi-task prefix-tuning with UNIFIEDSKG improves the performance on most tasks and show that T0, GPT-3, and Codex struggle in zero-shot and few-shot learning for **SKG**. **UNIFIEDSKG** also enables a series of controlled experiments on structured knowledge encoding variants across SKG tasks. We find that T5’s sensitivity to structured knowledge encoding variations varies across tasks. 

**UNIFIEDSKG** is easily extensible to more tasks. We encourage the researchers that want to promote their fantastic work to the community to make **pull request** to update their datasets, metrics, models! 



## Content

- [UnifiedSKG: A Unified Framework and Analysis for **S**tructured **K**nowledge **G**rounding](#unifiedskg--a-unified-framework-and-analysis-for---s--tructured---k--nowledge-grounding)
  * [Cloning this Repo](#cloning-this-repo)
  * [Dependency](#dependency)
    + [Sub-Modules](#sub-modules)
      - [~~TaPEx~~（we adopted its table processor into our code to do some changes）](#--tapex---we-adopted-its-table-processor-into-our-code-to-do-some-changes-)
      - [TaBERT](#tabert)
  * [To run](#to-run)
    + [Environment setup](#environment-setup)
    + [Training](#training)
    + [Deepspeed](#deepspeed)
    + With [wandb](https://wandb.ai/) report
  * [Introduction of each file](#introduction-of-each-file)
    + [configure](https://github.com/HKUNLP/UnifiedSKG/tree/master/configure)
    + [metrics](https://github.com/HKUNLP/UnifiedSKG/tree/master/metrics)
    + [models](https://github.com/HKUNLP/UnifiedSKG/tree/master/models)
    + [seq2seq_construction](https://github.com/HKUNLP/UnifiedSKG/tree/master/seq2seq_construction)
    + [third_party](https://github.com/HKUNLP/UnifiedSKG/tree/master/third_party)
    + [utils](https://github.com/HKUNLP/UnifiedSKG/tree/master/utils)
    + [train.py](#trainpy)
    + [Procedure](#procedure)
  * [The overview file structure of this Unified Framework](#the-overview-file-structure-of-this-unified-framework)
  * [How to unify a new task into the framework](#how-to-unify-a-new-task-into-the-framework)
  * [Ackonwledgement](#ackonwledgement)




## Cloning this Repo

In order to include third-party dependencies in this repository, make sure to clone recursively, e.g.:

```
git clone --recurse-submodules git@github.com:HKUNLP/UnifiedSKG.git
```



## Dependency

To establish the environment run this code in the shell (the third line is for CUDA11.1):

``````
conda env create -f py3.7pytorch1.8.yaml
conda activate py3.7pytorch1.8new
pip install datasets==1.14.0
# The following line to be replaced depending on your cuda version.
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
``````

That will create the environment `py3.7pytorch1.8` we used. 



### Sub-Modules

Some third party libraries stored in sub-modules need installation

#### ~~TaPEx~~（we adopted its table processor into our code to do some changes）

~~For TaPEx, you can run~~

```
cd third_party/table_pretraining/
pip install --editable ./
cd ../..
```


#### TaBERT

Run the following with the conda env activated and *after* installing the main dependencies for UniPSP:
``````
pip install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert fairseq==0.8.0 torch-scatter==1.3.2 ujson msgpack redis zmq h5py
``````

Then, navigate to the TaBERT directory and install it:

``````
cd third_party/tabert/
pip install --editable ./
cd ../..
``````

And if you add more modification to the env or more commands during you adding for unification, 
please note in the block below of this README:

``````
*add*me*
``````
we will compress them to create a docker environment in the end. 



## To run

### Environment setup

Configure [W & B](https://wandb.ai/) progress reporting support:

``````shell
export WANDB_ENTITY=YOUR_WANDB_USERNAME
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
``````

Environment setup

``````shell
conda activate py3.7pytorch1.8
export CUDA_LAUNCH_BLOCKING=1
``````

### Training

Resume training

``````shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_NAME=finetune_9tasks_01eval_bsz32_dist
nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_finetune.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 20 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1088 > $RUN_NAME.log 2>&1 &
``````

Training from scratch

``````shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_NAME=finetune_9tasks_01eval_bsz32_dist
nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_finetune.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 20 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1088 > $RUN_NAME.log 2>&1 &
``````

Single-task prefix-tuning

``````shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_NAME=post_spider_prefix_9tasks_1eval_bsz64_dist
nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --cfg META_TUNING/T5_prefix_freeze.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 100 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 20 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --load_weights_from output/prefix_9tasks_01eval_bsz32_dist/checkpoint-11500 --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1350 > $RUN_NAME.log 2>&1 &
``````

*And if you are using [tmux](https://en.wikipedia.org/wiki/Tmux), you can also open new sessions and sub-sessions to run seperate scripts on it, instead of using "nohup" and "&" to make the code run in background. 

### Deepspeed

Training from scratch

``````shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_NAME=finetune_9tasks_01eval_bsz32_fp16_deepspeed
nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 train.py --deepspeed deepspeed/ds_config_zero1.json --fp16 --cfg META_TUNING/T5_finetune.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 20 --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1440 > $RUN_NAME.log 2>&1 &
``````

### With [wandb](https://wandb.ai/) report

[wandb platform](https://docs.wandb.ai/) is a useful tool for us to track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings. We highly recommand using this platform to versialize, track and share results. 

To use that in our framework, you can change the env parameter WANDB_API_KEY, WANDB_PROJECT(you need to have a quick register on the wandb platform), and other task/training hyperparameter and use the shell below:



## Introduction of each file

### [configure](https://github.com/HKUNLP/UnifiedSKG/tree/master/configure)

Code for configuration of different tasks/settings,
more details see README in [./configure](https://github.com/HKUNLP/UnifiedSKG/tree/master/configure)

### [metrics](https://github.com/HKUNLP/UnifiedSKG/tree/master/metrics)
Code for evaluating the prediction of our model,
more details see README in [./metrics](https://github.com/HKUNLP/UnifiedSKG/tree/master/metrics)

### [models](https://github.com/HKUNLP/UnifiedSKG/tree/master/models)
Code for models(for now, we have seq2seq models(T5 and BART) and prompt-tuning models(prefix-tuning)

### [seq2seq_construction](https://github.com/HKUNLP/UnifiedSKG/tree/master/seq2seq_construction)
Code for evaluating the prediction of our model,
more details see README in [./seq2seq_construction](https://github.com/HKUNLP/UnifiedSKG/tree/master/seq2seq_construction)

### [third_party](https://github.com/HKUNLP/UnifiedSKG/tree/master/third_party)
packages from the third party for us to tmp store, and we will redirect them by git recursive deployment in the end. 

### [utils](https://github.com/HKUNLP/UnifiedSKG/tree/master/utils)
Code for some useful(or not) stuff, it contains:
- **configure.py**: The "args" data-structure for **parsing and store the config files** in ./configure. (and we are trying to change it 
into a more main-stream structure which also support read from the file and create nested config object.)
- **dataset.py**: Wrap the seq2seq dataset to tokenize the "seq_in" and "seq_out", since the trainer only support tokenized tensors of "seq_input" and "seq_output" as input
- **tool.py**: The tool to **get datasets, models and evaluators in a unified way**. 
- **trainer.py**: The **modified trainer version** of huggingface trainer class **for supporting meta-tuning**(we want get our training sampler under our control), 
**easier evaluation**(the metrics of huggingface's input format(numbers) is contradicted with ones of all official evaluations)
 and **further changes in this project**(for example, we want to feed more para in a model.forward function).
- **training_arguments.py**: The **customized wrapped training_arguments**.

### train.py
- together with the config file, act as the start and main-control of the experiment.

### Procedure
The working procedure of our work follows:
raw data(s) -> + seq2seq data(s) ("seq_in" and "seq_out") -> tokenized -> seq2seq_trainer -> predictions -> eval(s)

## The overview file structure of this Unified Framework
    .
    ├── configure                          # Code for configuration of different tasks/settings
    |   ├── META_TUNING # config files for running meta-tuning, some are father config controls which tasks to involve,
     some are child configs only have info of specific task
    │   ├── UNIFIED # config files for running the unified seq2seq tasks.
    │   ├── ...
    ├── metrics                       # Code for evaluating the prediction of our model
    │   ├── ...                     # each file contains a evaluator of the corresponding task
    ├── models                       # Code for models
    │   ├── prompt # the modified hugginface transformers, where we realized prefix-tuning in T5 and BART.
    |   ├── unified
    |       ├──finetune.py # model of the bare finetune
    |       ├──prefixtuning.py # model of the prefix-tuning, the prompt getting methods followed one of BART in original paper
    ├── seq2seq_construction                       # Code for wrap the raw data into seq_in and seq_out and add them
    │   ├── ... # check the README in the ./seq2seq_construction
    ├── tasks                       # Code for encoder-decoder architecture
    │   ├── ... # check the README in the ./tasks
    ├── third_party                       # packages from the third party
    │   ├── ...  # if you use any github repo from others, try to put them in this dir, and note the links in the .submodules 
    for us to make them easier to e added by the recursive clone of git.
    ├── utils                       # Code for some useful(or not) stuff
    │   ├── __init__.py             
    │   ├── configure.py           # the util for parsing the cfg file in ./configure, will get nested args object which is human friendly.
    │   ├── dataset.py               # wrap the seq2seq dataset constructed, tokenize the seq_in and seq_out for feed into the trainer.
    │   ├── tool.py         # Use the reflection to help loading model, seq2seq constructor and evaluator
    │   ├── trainer.py                  # we changed the original trainer into the EvaluationFriendlyTrainer in this file, for easier eval, also we controled the sequence of trainer to be in original order, 
    and add description in it, if you want make any modifications in forward of the models, you may need to change something in here.
    │   └── training_arguments.py              # wrapped training arguments for seq2seq
    ├── .gitignore              # use to ignored some tast or debug files in your desktop
    ├── .gitmodules           # use the recursive clone of the git, will be used to create files in ./third_party at last
    ├── py3.7pytorch1.8.yaml     # help you clone the anaconda env
    ├── README.md         # As you can see, is the README hhhhh
    └── train.py              # The start of the code, control the train, eval, test, store and log and 



## How to unify a new task into the framework

(README in ./tasks, ./seq2seq_construction, ./metrics, ./configure can also be useful)

- **step 1**, Add the "Loader" of raw data in ./tasks, (you can search in [huggingface dataset website](https://github.com/huggingface/datasets) firstly to find whether there is already a usable script, if not, that's great because you can be the contributor of both this project and huggingface community.

- **step 2**, Add the "Wrapper" which construct "seq_in"("user request input" & "structured knowledge input") and "seq_out" from and add to the raw_data for seq2seq unification.

- **step 3**, Add the evaluator(evaluate a task) in ./metrics. if use third_party repo, please add them into the ./third_party. dir

- ***step 3.5(optional)**, You can always add new models into the ./models/unified id you like.

- **step 4**, Add the config file to drive your task or all the tasks we have by finetune/prefix-tuning/multi-task-finetune/pretrain... or other ways. 

**And this is all for it ! =)**



## Ackonwledgement
Besides all authors in this project, we need to give thank to these people for their warming and kindly help for this project. We thank [Yifei Min](https://statistics.yale.edu/people/yifei-min) and [Libo Qin](http://ir.hit.edu.cn/~lbqin/) for early stage discussion. We thank [Qian Liu](https://siviltaram.github.io/) for [TAPEX code](https://github.com/microsoft/Table-Pretraining) and advice on Question Answering tasks. We thank [Ice Pasupat](https://ppasupat.github.io/) for reviewing this paper. We thank [wandb](https://wandb.ai/) for free logging and [OpenAI](https://openai.com/) for free Codex usage.





