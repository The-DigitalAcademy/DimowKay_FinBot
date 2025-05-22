from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.post("/api/generate")
async def proxy_generate(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=data
        )
        return response.json()