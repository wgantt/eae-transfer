This directory contains the prompt used to generate paraphrases of the FAMuS role questions and templates. We used GPT-4 (`gpt-4-0125-preview`) to generate these paraphrases with the following hyperparameters:

- `frequency_penalty`: 0.0
- `max_new_tokens`: 512
- `logit_bias`: null
- `n`: 1
- `presence_penalty`: 0.0
- `top_p`: 1.0
- `temperature`: 0.7

Examples of the user prompts are contained in `question_prompt.txt` and in `template_prompt.txt`. The only component of the prompt that changes between examples is the text following `Question:` (in the first case) and the text following `Template:` (in the second). NOTE: we did not use a system prompt when generating the paraphrases.