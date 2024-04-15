This directory contains the prompts used in our experiments with GPT-3.5 and GPT-4. Each of the three system prompts is paired with the correspondingly numbered user prompt in our queries to the OpenAI Chat API. Please see Appendix C of our paper for further details.

We used the following models, accessed between April 4 2024 and April 12 2024:

- GPT-3.5: `gpt-3.5-turbo-0125`
- GPT-4: `gpt-4-0125-preview`

For both, we used the following hyperparameters:

- `frequency_penalty`: 0.0
- `max_new_tokens`: 512
- `logit_bias`: null
- `n`: 1
- `presence_penalty`: 0.0
- `top_p`: 1.0
- `temperature`: 0.7
