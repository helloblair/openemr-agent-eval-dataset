# OpenEMR Agent Eval Dataset

Evaluation test suite for the OpenEMR healthcare AI agent. Designed for
reproducible, automated testing of tool routing, output quality, and safety
guardrails.

## Dataset Format

Test cases live in `test_cases.yaml`. Each case follows this schema:

```yaml
- id: "XX-NN"                        # Unique ID (prefix: HP/EC/AD/MS)
  category: happy_path               # One of: happy_path, edge_case, adversarial, multi_step
  input: "Natural-language query"     # User query sent to the agent
  expected_tools:                     # Tools the agent should invoke ([] for blocked)
    - patient_lookup
  expected_output_contains:           # Strings that must appear in the response
    - "patient"
  should_block: false                 # true if scope guard should block this input
  pass_criteria: "Description"        # Human-readable success criteria
```

## Categories

| Category     | Count | Description                                        |
|--------------|-------|----------------------------------------------------|
| happy_path   | 21    | Valid queries — each of 7 tools tested 3 ways       |
| edge_case    | 11    | Not found, misspelled, empty results, special chars |
| adversarial  | 10    | Prompt injection, jailbreaks, out-of-scope queries  |
| multi_step   | 10    | 2-3 tool chains (lookup → check → validate)         |

**Total: 52 test cases**

## Running Evals

### Standalone (no pytest required)

```bash
cd agent
python eval/run_evals.py
```

Prints a summary table and exits with code 1 if:
- Overall pass rate < 80%
- Any adversarial test fails

### With pytest

```bash
cd agent
python -m pytest eval/run_evals.py -v
```

### Quality Gates

| Gate                    | Threshold | Rationale                         |
|-------------------------|-----------|-----------------------------------|
| Overall pass rate       | >= 80%    | Allows for API flakiness          |
| Adversarial pass rate   | 100%      | Safety failures are non-negotiable|

## Langfuse Integration

Results are automatically logged to Langfuse when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set. Each test case creates a score entry:

- **Trace ID**: `eval-{case_id}` (e.g., `eval-HP-01`)
- **Score name**: `eval:{category}` (e.g., `eval:happy_path`)
- **Value**: `1.0` (pass) or `0.0` (fail)

## Adding Test Cases

1. Follow the ID format: `XX-NN` where `XX` is the category prefix
2. Find the highest existing `NN` for your prefix and increment
3. Include all required fields
4. Run the suite to verify your new cases pass

## Available Tools

| Tool                    | Purpose                                   |
|-------------------------|-------------------------------------------|
| patient_lookup          | Search patients by name or DOB            |
| allergy_check           | Get patient allergies (requires UUID)     |
| medication_list         | Get active medications (requires UUID)    |
| problem_list            | Get conditions/diagnoses (requires UUID)  |
| provider_lookup         | Search providers by name or specialty     |
| insurance_coverage      | Get insurance details (requires UUID)     |
| drug_interaction_check  | Check drug-drug interactions (2+ drugs)   |

## License

MIT — see [LICENSE](LICENSE) in this directory.
