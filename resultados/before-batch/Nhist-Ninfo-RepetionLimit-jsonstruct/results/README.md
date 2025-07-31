##command:

```script
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 100 -cc 100 --debug
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
"change features from the current_solution"
"Replace with one feature from the following list RCL, that is not in the current solution."
"Current solution and RCL in json format: {json.dumps(querry_json)}."
"Your answer must be a new solution with the feature replaced in a json format, e.g., '{{\"selected_features\":[featname1, featname2, ...]}}'. This format is required"
```

history: NO
Repetion Limit: 20

Notes: Resultado melhor com maior rcl e nao todas a features disponiveis para escolha. Usar um json parece melhorar o entendimento da LLM