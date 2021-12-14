#Configure part
## Why use config?
Since it is a very large project contains **a huge variety of tasks** related to the structure background knowledge,
 the unification of coding is also a challenge faced by all of us.

So we choose to use the configure file to drive specific task/setting. 
We utilise the config file to store the **specific info(which is also relatively fixed)** in the config files.
While the **flexible training arguments** to be stored in the huggingface's training_args.
These two kind of args have their own jobs. 

## What is it now? and want can i do?
Up to now, we divide all the configure files into two groups: 
The META_TUNING and UNIFIED. 
### UNIFIED, add more tasks to utilize:
It contains the config of specific task/setting which uses different models(prefix-tuning model/finetune model) we have now. 
And you can always **add config file with new settings** after you finished your adding in settings of this project,
Or add some new config parameter into the config file if you want to use them in anywhere of you task.

### META_TUNING, utilize the involved tasks:
It has some **main configs**(for example BART_prefix.cfg) which decides the tasks we want to involve in meta-tuning(meta-tuning means training all tasks together)
While other configs(**child configs**) are set for their task specific information. 
And yes,  you can always **add config file with new settings** after you finished your adding in settings of this project as the child config,
Or add some new config parameter into the config file if you want to use them in anywhere of you task.
Or involve new added tasks into the main config file.




