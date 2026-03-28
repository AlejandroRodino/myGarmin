"""
Strava OAuth2 authentication flow.

Steps:
1. Run this script
2. It opens your browser to authorize the app
3. You'll be redirected to localhost with a code
4. The script exchanges that code for access/refresh tokens
5. Tokens are saved to .tokens.json for reuse
"""

import json
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests

TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".tokens.json")


def load_env():
    """Load config from os.environ first, fall back to .env file (for Railway)."""
    # Check environment variables first (Railway sets these)
    if os.environ.get("STRAVA_CLIENT_ID") and os.environ.get("STRAVA_CLIENT_SECRET"):
        return {
            "STRAVA_CLIENT_ID": os.environ["STRAVA_CLIENT_ID"],
            "STRAVA_CLIENT_SECRET": os.environ["STRAVA_CLIENT_SECRET"],
        }

    # Fall back to .env file (local development)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            "Create a .env file with STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET, "
            "or set them as environment variables."
        )
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def bootstrap_tokens_from_env():
    """Create .tokens.json from STRAVA_REFRESH_TOKEN env var.

    On Railway's ephemeral filesystem, .tokens.json doesn't persist.
    This bootstraps it from environment on each deploy/restart.
    """
    refresh_token = os.environ.get("STRAVA_REFRESH_TOKEN")
    if not refresh_token:
        return False

    if os.path.exists(TOKEN_FILE):
        return True  # already exists

    env = load_env()
    print("Bootstrapping tokens from STRAVA_REFRESH_TOKEN env var...")
    tokens = refresh_access_token(
        env["STRAVA_CLIENT_ID"],
        env["STRAVA_CLIENT_SECRET"],
        refresh_token,
    )
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    print("Tokens bootstrapped successfully.")
    return True


def get_tokens_via_oauth(client_id, client_secret):
    """Run local OAuth flow: open browser, capture redirect, exchange code."""
    redirect_uri = "http://localhost:8642/callback"
    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&response_type=code"
        f"&redirect_uri={redirect_uri}"
        f"&scope=activity:read_all"
        f"&approval_prompt=auto"
    )

    captured_code = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            if "code" in query:
                captured_code["code"] = query["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Done! You can close this tab.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"No code received.")

        def log_message(self, *args):
            pass  # silence logs

    print(f"Opening browser for Strava authorization...")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8642), CallbackHandler)
    server.handle_request()  # wait for single callback

    if "code" not in captured_code:
        raise RuntimeError("Did not receive authorization code")

    # Exchange code for tokens
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": captured_code["code"],
            "grant_type": "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(client_id, client_secret, refresh_token):
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    return resp.json()


def get_access_token():
    """Return a valid access token, refreshing or re-authorizing as needed."""
    env = load_env()
    client_id = env["STRAVA_CLIENT_ID"]
    client_secret = env["STRAVA_CLIENT_SECRET"]

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            tokens = json.load(f)

        import time
        if tokens.get("expires_at", 0) > time.time():
            return tokens["access_token"]

        # Token expired — refresh
        print("Refreshing expired token...")
        tokens = refresh_access_token(
            client_id, client_secret, tokens["refresh_token"]
        )
        with open(TOKEN_FILE, "w") as f:
            json.dump(tokens, f, indent=2)
        return tokens["access_token"]

    # No tokens yet — full OAuth flow
    tokens = get_tokens_via_oauth(client_id, client_secret)
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    return tokens["access_token"]


if __name__ == "__main__":
    token = get_access_token()
    print(f"Access token obtained successfully.")
