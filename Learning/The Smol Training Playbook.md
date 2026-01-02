# The Smol Training Playbook

## Post-training

Pre-training is about forcing knowledge into weights, post-training is about sculpting that raw capability into something reliable and steerable.

### Eval

- Capability evals

1. Knowledge
2. Math
3. Code
4. Multilinguality

- Integrated task evals

1. Long context
2. Instruction following
3. Alignment
4. Tool calling

- Overfitting-prevention evals

- Internal evals

- Vibe evaluations and arenas

  “vibe testing” intermediate checkpoints (aka interacting with your model) is essential for uncovering subtle quirks in model behaviour that are not captured by eval scores.

#### A Few Hard-earned Lessons
- Use small subsets to accelerate evals during model development.
- For  reasoning models , strip the chain-of-thought from the  scored  output. This eliminates false positives and also directly impacts benchmarks like IFEval which penalise responses that violate constraints like “write a poem in under 50 words”.
- If an eval uses  LLM judges, pin the judge and version for apples-to-apples comparisons over time. Even better, use an open weight model so that the eval is reproducible even if a provider deprecates the judge model.
- Be wary of  contamination  in the base models.
- If possible, treat anything used during ablations as  validation , not  test.
- Always include a small set of  “vibe evals”  on your own data and tasks to catch overfitting to public suites.
- For evals with a small number of problems (typically less than ~2k), sample k times and report the avg@k accuracy.
- When implementing a new eval, make sure you can replicate the published result of a few models (within some error).
- When in doubt, always go back to the evaluation data, and notably inspect what you are prompting your models with

### SFT

In practice, SFT isn’t just the first step because it’s easy; it’s the step that consistently improves performance before anything more complex should be attempted. This is especially true when you are working with base models. 

- datasets

balance your data mixture in terms of tokens not examples

- chat template
- Baby baselines
  establish some “baby baselines”. These baselines aim to validate that the chat template does what you want and that the initial set of hyperparameters produce stable training.

  - full fine-tuning (FullFT) or parameter efficient methods like LoRA or QLoRA? As described in the wonderful blog post by Thinking Machines, LoRA can match FullFT under certain conditions.
  - What type of parallelism do you need? For small models or those trained with LoRA, you can usually get by with data parallel. For larger models you will need FSDP2 or DeepSpeed ZeRO-3 to shared the model weights and optimiser states. For models trained with long context, use methods like context parallelism.
  - Use kernels like FlashAttention and Liger if your hardware supports them.
  - Mask the loss to train only on assistant tokens.
  - Tune the learning rate; aside from the data, this is the most important factor that determines whether your model is “meh” vs “great”.
  - Pack the training samples and tune the sequence length to match the distribution of your data. This will dramatically speed up training.

- Continued pretraining when SFT alone isn’t enough.

### PO & RL

- datasets
  - Strong vs. weak
 
- algorithm
  - DPO
  - ORPO
