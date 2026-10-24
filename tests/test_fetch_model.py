import urllib.request

from scripts.fetch_model import SafeAssetRedirect


def test_github_credentials_are_not_forwarded_to_asset_storage():
    request = urllib.request.Request("https://api.github.com/repos/example/assets/1", headers={"Authorization": "Bearer test-value"})
    redirected = SafeAssetRedirect().redirect_request(request, None, 302, "Found", {}, "https://asset-storage.example/file.pt")
    assert not redirected.has_header("Authorization")
