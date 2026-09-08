# Developmental Learning v2.7

Advanced AI Agent v2.7 adds a developmental-learning layer on top of episodic memory, neural memory, the experience-policy network and v2.6 human-inspired cognitive consolidation.

## Stages

The agent advances through five learning stages based on durable XP earned from explicit feedback and deliberate practice:

`seed -> explorer -> apprentice -> practitioner -> specialist`

Stages change the learning strategy, not permissions or safety rules:

- **seed / explorer**: observe, compare and identify knowledge gaps.
- **apprentice**: practice and transfer prior knowledge through analogies.
- **practitioner**: generalize learned patterns and test them against new cases.
- **specialist**: use teach-back/refinement to expose weak assumptions.

## Curiosity

Curiosity is a bounded 0..1 prioritization signal calculated from:

- novelty of the current situation;
- low metacognitive confidence;
- lack of known cognitive patterns.

When curiosity exceeds the configured threshold, the agent can create a persistent learning goal automatically. Similar goals are merged rather than duplicated indefinitely.

Curiosity never grants permissions and never overrides user intent, safety rules or evidence.

## Learning goals

Goals keep:

- title and knowledge gap;
- priority and curiosity;
- current confidence;
- attempts and successful outcomes;
- open/mastered state.

By default, a goal needs four consistent successful outcomes and sufficient confidence before being considered mastered.

## Analogy transfer

Starting at the apprentice stage, high-curiosity situations can trigger an analogy between a consolidated cognitive memory and the new problem. The model must return both:

- the structural mapping;
- where the analogy stops being valid.

This reduces superficial analogy use and makes limitations explicit.

## Deliberate practice

`POST /v1/development/goals/{goal_id}/practice` creates one small practice exercise for a goal.

Conceptual exercises are generated and then separately graded. Algorithmic exercises may include a Python snippet.

### Python practice sandbox

The Python practice runner is deliberately conservative:

- static AST validation;
- allow-listed standard-library imports only;
- dangerous builtins rejected;
- dunder traversal rejected;
- isolated Python mode (`-I`);
- temporary working directory;
- stripped environment;
- timeout and output limits;
- POSIX CPU/memory/file-size caps where available.

This remains a **soft sandbox, not a VM/container security boundary**. Therefore execution is disabled by default:

```env
DEVELOPMENTAL_SANDBOX_EXECUTION_ENABLED=false
```

When disabled, generated Python can still be validated without executing it.

## Privacy

Queries that appear to contain passwords, API keys, bearer tokens, private keys, payment secrets or access/refresh tokens are not persisted as developmental goals. Analogy records receive the same obvious-secret filter.

## API

- `GET /v1/development/status`
- `GET /v1/development/goals`
- `POST /v1/development/goals`
- `POST /v1/development/goals/{goal_id}/practice`
- `GET /v1/development/practice`
- `POST /v1/development/sandbox/validate`

`GET /v1/learning/status` also includes the developmental profile alongside the experience policy, human-like learning and LoRA state.

## Autonomous behavior defaults

- Auto goals: **on**.
- Auto analogies: **on**, gated by stage and curiosity.
- Auto practice: **off** because it adds extra model calls.
- Sandbox execution: **off**.

Enable auto-practice only when the additional model usage is wanted:

```env
DEVELOPMENTAL_AUTO_PRACTICE=true
```

The system is human-inspired, not human consciousness. Stages, curiosity, reflection and confidence are computational mechanisms for better learning behavior.
