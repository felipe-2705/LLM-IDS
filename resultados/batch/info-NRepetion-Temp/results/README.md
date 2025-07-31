##command:

```script
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 5 -cb 40 -lb 10 --debug
```
temp: 0.7
avaliator: RandomFlorest

Constrution-prompt:
```
"Your goal is to generate Exactly {args.constructive_batch} unique feature sets (solutions), which solution must have EXACTLY {args.initial_solution} unique features that MUST be selected from RCL. "
"Input in json format: {json.dumps(query_json)}. "
"The description of each feature is {RCL_features_dict}."
"Output MUST be a single valid JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. This format is required. "
"Ensure the solutions are unique and selected feature set was not selected before. "
"Do NOT include any explanation, text, or Python code.. 
```
local search prompt:
```
"Your goal is to generate EXACTLY {batch_size} (NEVER MORE THAN THAT) unique feature sets (solutions), each with features that MUST be selected from RCL, based on modifying the best_solution."
"Do NOT reuse any solution from current_solutions."
"Features MUST be selected from RCL."
"Each solution must be at least the same size as best_solution"
"Inputs in json format: {json.dumps(querry_json)}."
"Output MUST be a single valid JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. Solutions list MUST BE exactly {batch_size} long. This format is required"
"Ensure the solutions are unique and selected feature set was not selected before."
"Do NOT include any explanation, text, or Python code.."
```

history: NO
Repetion Limit: 20

Notes: Temperatura nao parece ter afetado o resultado