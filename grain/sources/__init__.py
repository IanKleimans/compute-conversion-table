"""Public data sources: InferenceX output per GPU, GetDeploying rent, OpenRouter list prices."""

import json
import urllib.request

USER_AGENT = "grain-conversion-table/0.1"


def fetch(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: float = 120):
    return json.loads(fetch(url, timeout))
