"""Limit prediction body bytes before FastAPI parses the input JSON."""
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


MAX_PREDICTION_BODY_BYTES = 64 * 1024


class PredictionBodyLimit:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        if path != "/predict":
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                # Preserve FastAPI's existing disconnected-body handling. Returning
                # without invoking the inner app leaves outer middleware without a response.
                async def disconnected() -> dict:
                    return {"type": "http.disconnect"}

                await self.app(scope, disconnected, send)
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_PREDICTION_BODY_BYTES:
                await JSONResponse({"detail": "Prediction request body exceeds 64 KiB"},
                                   status_code=413)(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
