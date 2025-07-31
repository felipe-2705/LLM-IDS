##command:

```script
python main.py -a rf -rcl 50 -is 5 -pq 10 -lc 100 -cc 100
```
avaliator: RandomFlorest

Constrution-prompt:
```
"Select only {args.initial_solution} features from the following list: {', '.join(str(idx) for idx in RCL)}. "
"Return the selected features in a list format, e.g., selected_features=[featname1, featname2, ...]. This format is required"
"Ensure the selected features are unique and selected feature set was not selected before."
"only features names that exist in the provided list: {', '.join(str(idx) for idx in RCL)}."
```
local search prompt:
```
"change features from the current solution {new_solution_set}."
"Replace de choose feature with 1 feature from the following list: {', '.join(str(idx) for idx in RCL)}, that is not in the current solution."
"Return the selected features in a list format, e.g., selected_features=[featname1, featname2, ...]. This format is required"
"only features names that exist in the provided list: {', '.join(str(idx) for idx in RCL)}."
```

history: NO
Repetion Limit: 20

Notes: Resultados parecem nao melhorar com mais index juntos. LLM parece mais perdida