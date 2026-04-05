from pathlib import Path
import os
import base64
import json
import mimetypes

import yaml
from fastapi import FastAPI, HTTPException
from openai import AzureOpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Service")

PROMPTS_PATH = Path("/app/llm/prompts.yaml")
SCHEMAS_DIR = Path("/app/llm/schemas")
CONFIG_PATH = Path("/app/config.yaml")

prompts = {}
schemas = {}
client = None
CFG = None


class LLMRequest(BaseModel):
    prompt_name: str
    image_path: str
    prompt_vars: dict = Field(default_factory=dict)


def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"Image not found: {path}")

    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


@app.on_event("startup")
def startup():
    global prompts, schemas, client, CFG

    # load config + prompts
    CFG = yaml.safe_load(CONFIG_PATH.read_text())
    prompts = yaml.safe_load(PROMPTS_PATH.read_text())

    # preload all schemas
    for p in SCHEMAS_DIR.glob("*.json"):
        schemas[p.stem] = json.loads(p.read_text())

    # build client
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    print("============================")
    print("===      LLM SERVICE     ===")
    print("============================")
    print(f"Loaded prompts: {list(prompts.keys())}")
    print(f"Loaded schemas: {list(schemas.keys())}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: LLMRequest):
    if req.prompt_name not in prompts:
        raise HTTPException(404, "Prompt not found")

    if req.prompt_name not in schemas:
        raise HTTPException(404, "Schema not found")

    prompt = prompts[req.prompt_name]
    system = prompt["system"]
    user = prompt.get("user_template", "").format(**req.prompt_vars)

    image = image_to_data_url(req.image_path)

    res = client.responses.create(
        model=CFG["llm"]["deployment"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": user},
                {"type": "input_image", "image_url": image},
            ]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": req.prompt_name,
                "strict": True,
                "schema": schemas[req.prompt_name],
            }
        },
        max_output_tokens=CFG["llm"]["max_completion_tokens"],
    )

    return {"response": json.loads(res.output_text)}