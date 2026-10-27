"""Optional shared-secret boundary for the entire HTTP demo application."""
import base64
import binascii
from dataclasses import dataclass, field
import hmac
import os
from typing import Mapping

from starlette.responses import JSONResponse
from starlette.datastructures import State
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass(frozen=True)
class DemoAuthConfig:
    mode: str
    username: bytes = field(default=b"", repr=False)
    password: bytes = field(default=b"", repr=False)

    def authorization(self) -> str:
        return "Basic " + base64.b64encode(self.username + b":" + self.password).decode("ascii")


def load_demo_auth_config(environ: Mapping[str, str] | None = None) -> DemoAuthConfig:
    env = os.environ if environ is None else environ
    mode = env.get("DEMO_AUTH_MODE", "off")
    username = env.get("DEMO_AUTH_USERNAME", "")
    password = env.get("DEMO_AUTH_PASSWORD", "")
    if mode not in {"off", "basic"}:
        raise ValueError("Demo authentication mode must be off or basic")
    if mode == "off":
        if username or password:
            raise ValueError("Demo authentication credentials require basic mode")
        return DemoAuthConfig(mode)
    if not username.strip() or not password.strip():
        raise ValueError("Demo authentication requires a username and password")
    if ":" in username or any(ord(char) < 32 or ord(char) == 127 for char in username):
        raise ValueError("Demo authentication username contains invalid characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in password):
        raise ValueError("Demo authentication password contains invalid characters")
    return DemoAuthConfig(mode, username.encode("utf-8"), password.encode("utf-8"))


class DemoAccessGateMiddleware:
    def __init__(self, app: ASGIApp, app_state: State | None = None):
        self.app = app
        try:
            self.config = load_demo_auth_config()
        except ValueError:
            # Lazy construction precedes Starlette's lifespan error handling.
            # Report an explicit startup failure so Uvicorn auto cannot ignore it.
            self.config = None
        # The outer error handler needs the same validated mode, not credentials.
        if app_state is not None:
            app_state.demo_auth_mode = self.config.mode if self.config is not None else None

    def authenticated(self, scope: Scope) -> bool:
        values = [value for name, value in scope.get("headers", []) if name.lower() == b"authorization"]
        if len(values) != 1:
            return False
        scheme, separator, encoded = values[0].partition(b" ")
        if scheme.lower() != b"basic" or not separator:
            return False
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return False
        username, separator, password = decoded.partition(b":")
        username_matches = hmac.compare_digest(username, self.config.username)
        password_matches = hmac.compare_digest(password, self.config.password)
        return bool(separator) and username_matches and password_matches

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if self.config is None:
            message = "Demo authentication configuration is invalid"
            if scope["type"] == "lifespan":
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.failed", "message": message})
                raise ValueError(message)
            if scope["type"] == "http":
                await JSONResponse(status_code=503, content={"detail": message},
                    headers={"Cache-Control": "no-store"})(scope, receive, send)
                return
            raise ValueError(message)
        if scope["type"] != "http" or self.config.mode == "off":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path):]
        public_liveness = scope["method"] == "GET" and path == "/status"

        async def no_store(message):
            if message["type"] == "http.response.start":
                headers = [(name, value) for name, value in message.get("headers", [])
                           if name.lower() != b"cache-control"]
                message = {**message, "headers": headers + [(b"cache-control", b"no-store")]}
            await send(message)

        if not public_liveness and not self.authenticated(scope):
            await JSONResponse(status_code=401, content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": 'Basic realm="Fraud demo", charset="UTF-8"'})(scope, receive, no_store)
            return
        await self.app(scope, receive, no_store)
