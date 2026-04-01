# ----------------------------------------
# A) Use default object_id from config
# ----------------------------------------
curl -X POST http://localhost:8000/run_pipeline \
  -H "Content-Type: application/json" \
  -d '{}' \
  | jq


# ----------------------------------------
# B) Select object via LLM (natural query)
# ----------------------------------------
curl -X POST http://localhost:8000/run_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "object_query": "green box"
  }' \
  | jq