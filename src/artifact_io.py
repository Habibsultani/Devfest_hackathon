"""GitHub credential and redirect handling shared by CLI and application."""

import os
import subprocess
import urllib.parse
import urllib.request


class SafeAssetRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if (
            redirected is not None
            and urllib.parse.urlparse(request.full_url).hostname
            != urllib.parse.urlparse(new_url).hostname
        ):
            redirected.remove_header("Authorization")
        return redirected


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        env=environment,
    )
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not credentials.get("password"):
        raise RuntimeError(
            "GitHub credentials unavailable. Sign in to Git or set GITHUB_TOKEN "
            "with access to this repository."
        )
    return credentials["password"]
