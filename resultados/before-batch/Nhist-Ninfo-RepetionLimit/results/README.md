##command:

```script
python main.py -a rf -rcl 10 -is 5 -pq 10 -lc 10 -cc 10
```
avaliator: RandomFlorest

Constrution-prompt:
```
"Select only {args.initial_solution} features from the following list: {', '.join(str(idx) for idx in RCL)}. "
"Return the selected features in a list format, e.g., selected_features=[featname1, featname2, ...]. This format is required"
"Ensure the selected features are unique and selected feature set was not selected before."
```
local search prompt:
```
"change features from the current solution {new_solution_set}."
"Replace de choose feature with 1 feature from the following list: {', '.join(str(idx) for idx in RCL)}, that is not in the current solution."
"Return the selected features in a list format, e.g., selected_features=[featname1, featname2, ...]. This format is required"
```

history: NO

Repetion Limit: 20