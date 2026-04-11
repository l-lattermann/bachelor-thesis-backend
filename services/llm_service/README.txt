LLM SERVICE

Test Request:
curl -X POST http://localhost:8004/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "select_obj_id",
    "full_img_path": "/shared/pipeline_io/sam_segmented_output.png",
    "prompt_vars": {
      "object_query": "red box"
    }
  }'

Input:
- prompt_name: name of the prompt and schema
- prompt_vars: values for user_template
- full_img_path: optional full image
- zoomed_img_path: optional zoomed image

Output:
- response: JSON response following schema (e.g. object_id or grasp selection)

Notes:
- prompt and schema must both exist
- images are internally converted to base64 data URLs
- OPENAI_BASE_URL, AZURE_OPENAI_API_KEY and deployment must be set
- output is strictly schema-constrained