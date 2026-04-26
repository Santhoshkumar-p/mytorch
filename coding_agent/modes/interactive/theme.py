from __future__ import annotations

_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "background":  "#0d1117",
        "surface":     "#161b22",
        "primary":     "#58a6ff",
        "secondary":   "#3fb950",
        "foreground":  "#e6edf3",
        "muted":       "#8b949e",
        "error":       "#f85149",
        "warning":     "#d29922",
        "border":      "#30363d",
    },
    "light": {
        "background":  "#ffffff",
        "surface":     "#f6f8fa",
        "primary":     "#0969da",
        "secondary":   "#1a7f37",
        "foreground":  "#1f2328",
        "muted":       "#636c76",
        "error":       "#cf222e",
        "warning":     "#9a6700",
        "border":      "#d0d7de",
    },
}
_THEMES["default"] = _THEMES["dark"]


def get_theme_vars(name: str | None) -> dict[str, str]:
    return _THEMES.get((name or "default").lower(), _THEMES["default"])
