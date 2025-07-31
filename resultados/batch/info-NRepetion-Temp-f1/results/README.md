##command:

```script
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 5 -cb 20 -lb 20 --debug
```
temp: 0.7
avaliator: RandomFlorest

Constrution-prompt:
```
"Your goal is to generate Exactly {args.constructive_batch} unique feature sets (solutions), which solution must have EXACTLY {args.initial_solution} unique features that MUST be selected from RCL. "
"Input in json format: {json.dumps(query_json)}. "
"The description of each feature is {RCL_features_dict}. "
"Output MUST be a single valid JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. This format is required. "
"Ensure the solutions are unique (Feature order doesnt make a new solutions. [1,2] and [2,1] are same solution) and selected feature set was not selected before. "
"Do NOT include any explanation, text, or Python code.. "
```
local search prompt:
```
"Your goal is to generate EXACTLY {batch_size} (NEVER MORE THAN THAT) unique feature sets (solutions), each with features that MUST be selected from RCL, based on modifying the best_solution to improve F1-score. "
"Do NOT reuse any solution from current_solutions. current_solutions has values from previous iterations and its F1-Score. "
"Features MUST be selected from RCL. "
"Each solution must be at least the same size as best_solution. "
"The input has tuples with the solution and its F1-Score: [solution,F1-score]. "
"Inputs in json format: {json.dumps(querry_json)}."
"Output MUST be a single VALID JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. Solutions list MUST BE exactly {batch_size} long. This format is required. "
"JSON sintax MUST be correct. "
"Ensure the solutions are unique and selected feature set was not selected before. "
"Do NOT include any explanation, F1-score, text, or Python code.."
```

history: LOCAL
Repetion Limit: NO

Notes: incluir o F1 score nao parece ter melhorado o resultado. Adicionar F1 score aumento os erros da LLM ao responder, tentando incluir um F1 score na resposta mesmo nao solicitado, errando na sintax da resposta algumas vezes sendo necessario ressaltar a necessidade de um json valid e nao incluir nada alem do solicitado. 