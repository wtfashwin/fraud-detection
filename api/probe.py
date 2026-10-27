"""Silent readiness probe; credentials stay in the process, never its argv."""
from http.client import HTTPException
import sys
import urllib.request

from api.demo_auth import load_demo_auth_config


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    if len(sys.argv) != 1:
        return 1  # No configurable destination or credential arguments.
    try:
        config = load_demo_auth_config()
        headers = {"Authorization": config.authorization()} if config.mode == "basic" else {}
        request = urllib.request.Request("http://127.0.0.1:8000/ready", headers=headers)
        # Ignore proxy environment and refuse redirects, including local ones.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=3) as response:
            return 0 if response.status == 200 else 1
    except (OSError, ValueError, HTTPException):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
