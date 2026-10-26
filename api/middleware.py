"""Bound prediction bodies before FastAPI reads or parses their JSON."""
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_PREDICTION_BODY_BYTES = 65536


class PredictionBodyLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path):]
        if scope["method"] != "POST" or path != "/predict":
            await self.app(scope, receive, send)
            return

        too_large = JSONResponse(status_code=413, content={
            "detail": f"Prediction body must not exceed {MAX_PREDICTION_BODY_BYTES} bytes"})
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared_size = int(value)
                except ValueError:
                    continue  # The streamed byte count remains authoritative.
                if declared_size > MAX_PREDICTION_BODY_BYTES:
                    await too_large(scope, receive, send)
                    return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                await JSONResponse(status_code=400, content={"detail": "Incomplete request body"})(
                    scope, receive, send)
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_PREDICTION_BODY_BYTES:
                await too_large(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        body_sent = False

        async def replay_body():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_body, send)
