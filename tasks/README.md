# Read in the raw data item in here.
In this file we write python files just for read in the raw data items.

we followed the [huggingface datasets package](https://huggingface.co/datasets). 
## How to add
When you are adding the tasks in this project, you **can firstly have a check on the official website** of datasets package.
Some datasets loader scripts are already written by others so you can copy that to our project.

If there's no such python file or we need to extract more information than their implements do,
For example, we need to add results queried by the sql for wikisql weakly supervised setting.
Your code can make a **contribution in datasets community**.

If you are not familiar with this package and don't know how to write the loader in their format, 
Please read the other python file for example, spider.py or wikisql.py.


