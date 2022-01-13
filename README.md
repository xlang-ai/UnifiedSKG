# UnifiedSKG:books:: Unifying Structured Knowledge Grounding with Text-to-Text Language Models

<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/HKUNLP/UnifiedSKG?color=green">
        <img src="https://img.shields.io/github/last-commit/HKUNLP/UnifiedSKG?color=green">
    </a>
    <a href="https://colab.research.google.com/drive/1f9yTXC3GpSyRJOjzsKceG_bhk-Cw71Ga#scrollTo=r_3-DN0SvC97">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <br/>
</p>

Code for paper [UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models](https://taoyds.github.io/assets/publications/unifiedskg.pdf). Please refer to our [project page](https://unifiedskg.com/) for up-to-date related resources (e.g., papers, code, tools, tutorials) in Structured Knowledge Grounding. 

<img src="pics/logos.png" align="middle" width="98%">

**S**tructured **k**nowledge **g**rounding (**SKG**) leverages structured knowledge to complete user requests, such as semantic parsing over databases and question answering over knowledge bases. Since the inputs and outputs of SKG tasks are heterogeneous, they were historically studied in separate by different communities,  which limits systematic and compatible research on SKG. In this paper, we overcome this limitation by proposing the **UNIFIEDSKG framework**, which unifies **21 SKG tasks** into the text-to-text format, aiming to promote systematic SKG research, instead of being exclusive to a single task, domain, or dataset. We show that large language models like T5, with simple modification when necessary, achieve **state-of-the-art performance on all 21 tasks**. **UNIFIEDSKG** facilitates the investigation of **multi-task, zero-shot, and few-shot learning**. We demonstrate that multi-task prefix-tuning with UNIFIEDSKG improves the performance on most tasks and show that T0, GPT-3, and Codex struggle in zero-shot and few-shot learning for **SKG**. **UNIFIEDSKG** also enables a series of controlled experiments on structured knowledge encoding variants across SKG tasks. We find that T5’s sensitivity to structured knowledge encoding variations varies across tasks. 

**UNIFIEDSKG** is easily extensible to more tasks. We encourage the researchers that want to promote their fantastic work to the community to make **pull request** to update their datasets, metrics, models! 

## Updates
- **2022-01-12**: We released our [code](https://github.com/HKUNLP/UnifiedSKG), [colab demo](https://colab.research.google.com/drive/1f9yTXC3GpSyRJOjzsKceG_bhk-Cw71Ga#scrollTo=r_3-DN0SvC97), [weights](https://huggingface.co/hkunlp) and [home page](https://unifiedskg.com). Check it out!

## Content

- [UnifiedSKG: A Unified Framework and Analysis for **S**tructured **K**nowledge **G**rounding](#unifiedskg--a-unified-framework-and-analysis-for---s--tructured---k--nowledge-grounding)
  * [Cloning this Repo](#cloning-this-repo)
  * [Dependencies](#dependencies)
  * [Usage](#usage)
    + [Environment setup](#environment-setup)
    + [Wandb setup](#wandb-setup)
    + [Training](#training)
    + [Load Weight](#load-weight)
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

## Dependencies

To establish the environment run this code in the shell (the third line is for CUDA11.1):

``````
conda env create -f py3.7pytorch1.8.yaml
conda activate py3.7pytorch1.8new
pip install datasets==1.14.0
# The following line to be replaced depending on your cuda version.
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
``````

That will create the environment `py3.7pytorch1.8` we used. 

<!--
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

-->

## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate py3.7pytorch1.8
``````

### WandB setup

Setup [WandB](https://wandb.ai/) for logging (registration needed):
``````shell
export WANDB_ENTITY=YOUR_WANDB_USERNAME
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
``````

### Training

T5-base finetuning on WikiTQ (4 GPUs, 128 effective batch size)
``````shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_base_finetune_wikitq.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
``````
If you want to resume training, remove the ``--overwrite_output_dir`` flag from the above command:
``````shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_base_finetune_wikitq.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_finetune_wikitq --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
``````

T5-base prefix-tuning on WikiTQ (4 GPUs, 128 effective batch size)
``````shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_base_prefix_wikitq.cfg --run_name T5_base_prefix_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_prefix_wikitq --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
``````

T5-3b finetuning on WikiTQ (8 GPUs, 128 effective batch size)
``````shell
deepspeed train.py --deepspeed deepspeed/ds_config_zero2.json --seed 2 --cfg Salesforce/T5_3b_finetune_wikitq.cfg --run_name T5_3b_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 50 --adafactor false --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_3b_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
``````

### Load Weight
See <a href="https://colab.research.google.com/drive/1f9yTXC3GpSyRJOjzsKceG_bhk-Cw71Ga#scrollTo=r_3-DN0SvC97">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
 
<!--
## Introduction of each file

### [configure](https://github.com/HKUNLP/UnifiedSKG/tree/master/configure)

Code for configuration of different tasks/settings, more details see README in [./configure](https://github.com/HKUNLP/UnifiedSKG/tree/master/configure)

### [metrics](https://github.com/HKUNLP/UnifiedSKG/tree/master/metrics)
Code for evaluating the prediction of our model, more details see README in [./metrics](https://github.com/HKUNLP/UnifiedSKG/tree/master/metrics)

### [models](https://github.com/HKUNLP/UnifiedSKG/tree/master/models)
Code for models(for now, we have seq2seq models(T5 and BART) and prompt-tuning models(prefix-tuning)

### [seq2seq_construction](https://github.com/HKUNLP/UnifiedSKG/tree/master/seq2seq_construction)
Code for evaluating the prediction of our model, more details see README in [./seq2seq_construction](https://github.com/HKUNLP/UnifiedSKG/tree/master/seq2seq_construction)

### [third_party](https://github.com/HKUNLP/UnifiedSKG/tree/master/third_party)
Packages from the third party for us to tmp store, and we will redirect them by git recursive deployment in the end. 

### [utils](https://github.com/HKUNLP/UnifiedSKG/tree/master/utils)
Code for some useful(or not) stuff, it contains:
- **configure.py**: The "args" data-structure for **parsing and store the config files** in ./configure. (and we are trying to change it 
into a more main-stream structure which also support read from the file and create nested config object.)
- **dataset.py**: Wrap the seq2seq dataset to tokenize the "seq_in" and "seq_out", since the trainer only support tokenized tensors of "seq_input" and "seq_output" as input
- **tool.py**: The tool to **get datasets, models and evaluators in a unified way**. 
- **trainer.py**: The **modified trainer version** of huggingface trainer class **for supporting meta-tuning**(we want get our training sampler under our control), 
**easier evaluation**(the metrics of huggingface's input format(numbers) is contradicted with ones of all official evaluations) and **further changes in this project**(for example, we want to feed more para in a model.forward function).
- **training_arguments.py**: The **customized wrapped training_arguments**.

### train.py
- together with the config file, act as the start and main-control of the experiment.

### Procedure
The working procedure of our work follows:
raw data(s) -> + seq2seq data(s) ("seq_in" and "seq_out") -> tokenized -> seq2seq_trainer -> predictions -> eval(s)
-->

## The overview file structure of this Unified Framework
    .
    ├── configure                          # Code for configuration of different tasks/settings
    │   ├── META_TUNING # meta config for each task, controls how we construct it to seq2seq data
    │   └── Salesforce # configs of our experiments settings, also thanks to ElementAI, Rui Zhang
    │
    ├── metrics                       # code for evaluating the prediction of our model
    │   └── ...                     # each file contains a evaluator of the corresponding task
    ├── models                       # code for models
    │   ├── adapter # the overwritten hugginface transformers, where we realized adapter in T5 and BART.
    │   ├── prompt # the overwritten hugginface transformers, where we realized prefix-tuning in T5 and BART.
    │   └── unified
    │           ├── base.py # base model for better push to huggingface arxiv(so called PushToHubFriendlyModel)
    │           ├── finetune.py # model of the bare finetune
    │           ├── adaptertuning.py # model of adapter-tuning,
    │           ├── prefixtuning.py # model of the prefix-tuning, the prompt getting methods followed one of BART in original paper
    │           └── multitask_prefixtuning.py # model of the mult-task prefix-tuning, support load from different single task prefix and make interaction on them(separate, concat...), but we didn't get satisfying result by doing so. Multi-task in our paper analysis section is SPoT-like.
    │
    ├── seq2seq_construction                       # Code for wrap the raw data into seq_in and seq_out and add them
    │    └──  ... # check the README in the ./seq2seq_construction
    │
    ├── tasks                       # Code for encoder-decoder architecture
    │    └──  ... # check the README in the ./tasks
    │
    ├── third_party                       # packages from the third party
    │    └──  ...  # if you use any github repo from others, try to put them in this dir, and note the links in the .submodules for us to make them easier to added by the recursive clone of git.
    │
    ├── utils                       # Code for some useful(or not) stuff
    │       ├── processor           # adopted from Tapex, processor for handling structured knowledge like table(truncation, linearized etc.) 
    │       ├── configure.py           # the util for parsing the cfg file in ./configure, will get nested args object which is human friendly.
    │       ├── dataset.py               # wrap the seq2seq dataset constructed, tokenize the seq_in and seq_out for feed into the trainer.
    │       ├── tool.py         # Use the reflection to help loading model, seq2seq constructor and evaluator
    │       └── trainer.py                  # we changed the original trainer into the EvaluationFriendlyTrainer in this file, for easier eval, also we controled the sequence of trainer to be in original order, and add description in it, if you want make any modifications in forward of the models, you may need to change something in here.
    │       └── training_arguments.py              # wrapped training arguments for seq2seq
    │
    ├── .gitignore              # use to ignored some tast or debug files in your desktop
    ├── .gitmodules           # use the recursive clone of the git, will be used to create files in ./third_party at last
    ├── py3.7pytorch1.8.yaml     # help you clone the anaconda env
    ├── README.md         # As you can see, is the README hhhhh
    └── train.py              # The start of the code, control the train, eval, test, store and log and 



## How to unify a new task into the framework

(README in ./tasks, ./seq2seq_construction, ./metrics, ./configure can also be useful)

- **step 1**, Add the "Loader" of raw data in ./tasks, (you can search in [huggingface dataset website](https://github.com/huggingface/datasets) firstly to find whether there is already a usable script, if not, that's great because you can be the contributor of both this project and huggingface community.

- **step 2**, Add the "Wrapper" which construct "seq_in"("user request input" & "structured knowledge input") and "seq_out" from and add to the raw_data for seq2seq unification.

- **step 3**, Add the "Evaluator"(for task) in ./metrics. if any third_party repo are used, please add them into [.gitmodules](https://git-scm.com/docs/gitmodules).

- **step 3.5(optional)**, You can always add new "Model" into the ./models/ if you like, change the path in config files to drive new model.

- **step 4**, Add the "Config" file to drive your task or all the tasks we have by finetune/multi-task-finetune/pretrain/prefix-tuning/multi-task-prefix-tuning... or other ways. 

**And this is all for it ! =)**

## Contributors
<a href="https://github.com/Timothyxxx">  <img src="https://avatars.githubusercontent.com/u/47296835?v=4"  width="50" /></a> 
<a href="https://github.com/ChenWu98"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a> 
<a href="https://github.com/Impavidity">  <img src="https://avatars.githubusercontent.com/u/9245607?v=4"  width="50" /></a> 
<a href="https://github.com/michiyasunaga"><img src="https://avatars.githubusercontent.com/u/25519127?v=4"  width="50" /></a>
<a href="https://github.com/cascadianblue"><img src="https://avatars.githubusercontent.com/u/6520892?v=4"  width="50" /></a>
<a href="https://github.com/chengzu-li"><img src="https://avatars.githubusercontent.com/u/69832207?v=4"  width="50" /></a>
<a href="https://github.com/jasonwu0731"><img src="https://avatars.githubusercontent.com/u/14951842?v=4"  width="50" /></a>
<a href="https://github.com/HKUNLP/UnifiedSKG/pulls"><img src="https://blog.simtics.com/wp-content/uploads/2016/03/you.jpg"  width="50" /></a>





