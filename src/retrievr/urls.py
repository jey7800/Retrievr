"""URL helpers: normalization (for dedup and matching) and domain checks."""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}
_PATH_SAFE = "/:@!$&'()*+,;=-._~"
_TRACKING_PREFIXES = ("utm_",)


def normalize_url(url: str) -> str:
    """Return a canonical form of ``url`` used as an identity key.

    Lowercases scheme and host, drops default ports and fragments, normalizes
    percent-encoding, removes ``utm_*`` parameters and sorts the query, so that
    ``?b=1&a[x]`` and ``?a%5Bx%5D=&b=1`` compare equal.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host if port is None or _DEFAULT_PORTS.get(scheme) == port else f"{host}:{port}"
    path = quote(unquote(_remove_dot_segments(parts.path or "/")), safe=_PATH_SAFE)
    pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(pairs), quote_via=quote)
    return urlunsplit((scheme, netloc, path, query, ""))


def resolve_url(href: str, base_url: str) -> str:
    """Resolve ``href`` against ``base_url`` and drop the fragment."""
    absolute = urljoin(base_url, href.strip())
    parts = urlsplit(absolute)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), "", "", ""))


def is_allowed(url: str, allowed_domains: frozenset[str] | set[str] | list[str]) -> bool:
    """True if the URL is http(s) and its host is one of ``allowed_domains`` exactly."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        return False
    return host_of(url) in {d.lower() for d in allowed_domains}


def path_extension(url: str) -> str:
    """Lowercased file extension of the URL path, without the dot ('' if none)."""
    last = unquote(urlsplit(url).path).rsplit("/", 1)[-1]
    return last.rsplit(".", 1)[-1].lower() if "." in last else ""


def _remove_dot_segments(path: str) -> str:
    output: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if len(output) > 1:
                output.pop()
        elif segment != ".":
            output.append(segment)
    result = "/".join(output)
    if path.endswith(("/.", "/..")):
        result += "/"
    return result or "/"
