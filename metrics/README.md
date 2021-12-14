# Add new evaluator in here! 

Add the evaluator(evaluate a task) in here (./metrics).

Each file is for a specific setting, the evaluator in each file **takes predictions(a list of predictions)
 and golds(a list of the gold data item, each data item is a dict) and return a dict as the evaluation result.**
 
 
 **if you use third_party repo in constructing the evaluator**, please add them into the **./third_party** dir, and note down their 
 link in .gitsubmodule, for us to make them recursive-able by git in the end.
 
 After all evaluators all written, we will handle the issue of how to reuse some of the code. So don't worry about that for now.
