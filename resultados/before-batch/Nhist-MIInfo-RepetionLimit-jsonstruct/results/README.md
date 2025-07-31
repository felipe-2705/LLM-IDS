##command:

```script
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 100 -cc 100
```
avaliator: RandomFlorest

Constrution-prompt:
```
"I want you to Select only {args.initial_solution} best features from list RCL in json{json.dumps(query_json)}."
"Your answer must be selected features in a json format, e.g., '{{\"selected_features\":[featname1, featname2, ...]}}'. This format is required"
"Ensure the selected features are unique and selected feature set was not selected before."
```
local search prompt:
```
"I want you to change features from the current_solution to improve the F1-Score."
"Replace it with one feature from the following list RCL, that is not in the current solution."
"RCL list is order by Mutual Information with first feature being the most informative."
"Current solution and RCL in json format: {json.dumps(querry_json)}."
"Your answer must be a new solution with the feature replaced in a json format, e.g., '{{\"selected_features\":[featname1, featname2, ...]}}'. This format is required"
                
```

history: NO
Repetion Limit: 20

Notes: Informa que RCL é ordenado pelo f1 score nao parece ter causado nenhuma melhora, repetiçoes totais cairam mas nao de forma significativa. A LLM as vezes entendia a informação, as vezes calculava ela mesma o MI.