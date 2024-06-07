from fastapi import Request, Response, FastAPI
import uvicorn, logging


async def launch_http_server(port, handler):
    app = FastAPI(docs_url=None, redoc_url=None)

    async def step(request: Request):
        body = await request.body()
        return Response(content=await handler(body))

    app.add_api_route("/step", step, methods=["POST"])

    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, 
        timeout_keep_alive=60 * 5, log_level=logging.WARN)
    await uvicorn.Server(config).serve()