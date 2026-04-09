from pathlib import Path
import os
import base64
import json
import mimetypes

import yaml
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Service")

CONFIG_PATH = Path("/app/config.yaml")

prompts = {}
schemas = {}
client = None
LLM_CFG = None
MODEL = None


class LLMRequest(BaseModel):
    prompt_name: str
    full_img_path: str | None = None
    zoomed_img_path: str | None = None
    prompt_vars: dict = Field(default_factory=dict)


def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@app.on_event("startup")
def startup():
    global prompts, schemas, client, LLM_CFG, MODEL

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    LLM_CFG = cfg["llm"]

    prompts = yaml.safe_load(Path(LLM_CFG["prompt_path"]).read_text())
    schemas = {
        p.stem: json.loads(p.read_text())
        for p in Path(LLM_CFG["schema_path"]).glob("*.json")
    }

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT") or LLM_CFG["deployment"]

    if not base_url:
        raise RuntimeError("Missing OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY")

    client = OpenAI(base_url=base_url, api_key=api_key)

    print("=== LLM SERVICE ===")
    print(f"Base URL: {base_url}")
    print(f"Model: {MODEL}")
    print(f"Prompts: {list(prompts)}")
    print(f"Schemas: {list(schemas)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: LLMRequest):
    if req.prompt_name not in prompts:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if req.prompt_name not in schemas:
        raise HTTPException(status_code=404, detail="Schema not found")

    prompt = prompts[req.prompt_name]
    system = prompt["system"]
    user = prompt.get("user_template", "").format(**req.prompt_vars)

    user_content = [{"type": "input_text", "text": user}]

    if req.zoomed_img_path:
        user_content.append({
            "type": "input_image",
            "image_url": image_to_data_url(req.zoomed_img_path),
        })

    if req.full_img_path:
        user_content.append({
            "type": "input_image",
            "image_url": image_to_data_url(req.full_img_path),
        })

    try:
        res = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": req.prompt_name,
                    "strict": True,
                    "schema": schemas[req.prompt_name],
                }
            },
            max_output_tokens=LLM_CFG["max_completion_tokens"],
        )
        return {"response": json.loads(res.output_text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")