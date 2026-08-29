# Project Instructions

## Subagent model selection

When delegating work via the Agent tool, assess how complex the task actually is before choosing a model rather than always defaulting to a higher-tier model. For simple, mechanical work — single-file lookups, straightforward searches, short summaries, or other tasks that don't require multi-step reasoning — pass `model: "haiku"` explicitly on the Agent call. Reserve Sonnet or Opus for subagent tasks that genuinely need judgment, synthesis, or multi-step reasoning.
