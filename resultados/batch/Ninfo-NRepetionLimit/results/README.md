##command:

```script
python main.py -a rf -rcl 20 -is 5 -pq 10 -lc 5 -cb 20 -lb 10 --debug
```
avaliator: RandomFlorest

System behavior:
{
                        "role": "system",
                        "content": (
                            "You are a strict assistant that only replies in the exact format requested. "
                            "Never add extra explanation, code formatting, or comments. "
                            "Only return valid JSON if asked, and always respect numerical limits (e.g., MAX 10 items). "
                            "If the user requests something structured, follow the structure precisely."
                        )
                        }

Constrution-prompt:
```
"Your goal is to generate Exactly {args.constructive_batch} unique feature sets (solutions), which solution must have EXACTLY {args.initial_solution} unique features that MUST be selected from RCL"
"Input in json format: {json.dumps(query_json)}."
"Output MUST be a single valid JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. This format is required"
"Ensure the solutions are unique and selected feature set was not selected before."
"Do NOT include any explanation, text, or Python code.."
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

history: LOCAL
Repetion Limit: NO



Notes: Com Batch a LLM parece nao respeitar as regras gerando listas muito maiores que as solicitadas e excedendo o limit de tokens. A repetição parece melhorar nos primeiros sets mas depois se torna um problema por isso reduzi as iterações locais. Muitas soluções invalidas sao geradas por isso foi necessario expecificar o comportamento da llm reduzindo esse numero para 0.