LLM SERVICE – README

Overview
--------
The LLM service provides a simple API to run vision-based reasoning using Azure OpenAI.
It takes:
- a prompt (defined in prompts.yaml)
- an image (path to file)
- optional variables

It returns:
- structured JSON output (validated via schema)


How to Call
-----------
curl -X POST http://localhost:8004/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "select_grasp",
    "image_path": "/shared/pipeline_io/cgn_output.png"
  }'


Folder Structure
----------------
/app/llm/
  ├── prompts.yaml        # all prompts
  ├── schemas/
  │     ├── select_grasp.json
  │     ├── select_object_id.json
  └── ...


Prompts
-------
- Stored in prompts.yaml
- Each prompt has:
  - system: instructions
  - optional user_template

Example:
select_grasp:
  system: |
    ...
  user_template: |
    User request: {object_query}

Rules:
- Keep prompts strict and unambiguous
- Avoid long text → models perform better with clear constraints
- Always align prompt output with schema


Schemas
-------
- Stored in /schemas as JSON
- Enforced via OpenAI structured output (json_schema)

Example:
{
  "type": "object",
  "properties": {
    "grasp_id": { "type": ["integer", "null"] },
    "reason": { "type": "string" }
  },
  "required": ["grasp_id", "reason"],
  "additionalProperties": false
}

Rules:
- Keep schemas minimal
- Always set "additionalProperties": false
- Match schema EXACTLY with prompt output


Important Things to Be Aware Of
------------------------------
1. Strict JSON mode
   - Model MUST follow schema → otherwise request fails
   - Bad prompts → invalid JSON → runtime error

2. Image path
   - Must exist inside container
   - Typically shared via Docker volume (/shared/...)

3. Prompt ↔ Schema coupling
   - If you change one → update the other
   - Most common bug source

4. Determinism
   - Do not rely on free-text parsing
   - Always use structured output

5. Performance
   - Image encoding (base64) adds overhead
   - Keep images small if possible


How to Adjust Prompts
--------------------
- Edit prompts.yaml
- Focus on:
  - constraints ("ONLY return...")
  - decision criteria
  - edge cases (e.g. "if none → return null")

Good:
- clear rules
- short bullet constraints

Bad:
- vague wording
- too much explanation


How to Adjust Schemas
--------------------
- Edit JSON file in /schemas
- Keep:
  - simple types (int, string, bool)
  - minimal fields

Optional design:
- Use booleans for evaluation criteria
- Use one final decision field (e.g. grasp_id)


Typical Workflow
----------------
1. Adjust prompt in prompts.yaml
2. Adjust schema in /schemas/
3. Restart container
4. Test via curl
5. Debug:
   - check JSON parsing errors
   - check missing fields
   - check model hallucinations


Common Errors
-------------
- "Schema not found" → missing file
- "Invalid JSON output" → prompt mismatch
- "Image not found" → wrong path / volume
- "Missing env var" → Azure config not set