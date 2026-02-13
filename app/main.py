import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.models import NithinGenerateRequest, NithinGenerateResponse
from app.nithin_post_generator import get_nithin_generator


app = FastAPI(title="Nithin Kamath Post Generator")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/nithin/status")
async def get_nithin_status():
    generator = get_nithin_generator()
    return {"available": generator.is_available()}


@app.post("/api/nithin/generate", response_model=NithinGenerateResponse)
async def generate_nithin_post(request: NithinGenerateRequest):
    platform = request.platform.lower().strip()
    if platform not in {"x", "linkedin"}:
        raise HTTPException(status_code=400, detail="Platform must be 'x' or 'linkedin'")

    generator = get_nithin_generator()
    result = generator.generate(
        context=request.context,
        platform=platform,
        facts=request.facts,
        angle=request.angle,
        cta=request.cta,
        thread=request.thread if platform == "x" else False,
        variants=request.variants,
        max_chars=request.max_chars,
        allow_research=request.allow_research,
        research_query=request.research_query,
        auto_research=request.auto_research,
        proofread=request.proofread
    )

    return NithinGenerateResponse(
        text=result.text,
        warnings=result.warnings,
        metadata=result.metadata
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
