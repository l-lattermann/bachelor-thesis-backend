from pathlib import Path
import os
import base64
import mimetypes
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AzureOpenAI

app = FastAPI(title="LLM Service")

PROMPTS_PATH = Path("/app/prompts.yaml")
CONFIG_PATH = Path("/app/config.yaml")

prompts = None
client = None
CFG = None


class LLMRequest(BaseModel):
    prompt_name: str
    image_path: str
    prompt_vars: dict = Field(default_factory=dict)


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    return load_yaml(CONFIG_PATH)


def get_env(name: str, required: bool = False):
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT", required=True),
        api_key=get_env("AZURE_OPENAI_API_KEY", required=True),
        api_version=get_env("AZURE_OPENAI_API_VERSION", required=True),
    )


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"

    with path.open("rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_b64}"


@app.on_event("startup")
def startup_event():
    global prompts, client, CFG

    CFG = load_config()
    prompts = load_yaml(PROMPTS_PATH)
    client = build_client()

    print("============================")
    print("===      LLM SERVICE     ===")
    print("============================")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "llm"}


@app.post("/generate")
def generate(req: LLMRequest) -> dict:
    try:
        print(f"[LLM] Request received: prompt_name={req.prompt_name}")

        if req.prompt_name not in prompts:
            raise HTTPException(status_code=404, detail=f"Prompt not found: {req.prompt_name}")

        prompt_cfg = prompts[req.prompt_name]
        system_prompt = prompt_cfg["system"]
        user_text = prompt_cfg.get("user_template", "").format(**req.prompt_vars)

        print(f"[LLM] Loading image: {req.image_path}")
        image_data_url = image_to_data_url(req.image_path)
        print("[LLM] Image loaded and encoded")

        print("[LLM] Sending request to Azure OpenAI...")
        response = client.responses.create(
            model=CFG["llm"]["deployment"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
            max_output_tokens=CFG["llm"]["max_completion_tokens"],
        )
        print("[LLM] Azure response received")

        return {
            "status": "ok",
            "service": "llm",
            "prompt_name": req.prompt_name,
            "response": response.output_text,
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing prompt variable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[LLM] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))