# seq2seq_construction---Wrap the raw data into 'seq_in' and 'seq_out' for our unification 
In this file we write python files for "wrap" the raw data into a seq2seq format.

We receive the **DatasetDict**(which contains train, dev, (test)) from the raw data split we got from the last step.
And wrap them to form **three/two Dataset**(torch.nn.data.Dataset) for feeding to the [huggingface trainer](https://huggingface.co/transformers/main_classes/trainer.html).

The args are from the args we set in the config file. Feel free to open your config files 
in configure file and add you own args if you need it in the seq2seq_construction.
 
## How to add
Baobao highly recommand guys to read the other python file for example, spider.py or wikisql.py.


