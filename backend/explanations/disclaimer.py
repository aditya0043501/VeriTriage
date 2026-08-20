"""
Disclaimer component for the patient explanation layer.

This text must be displayed visibly alongside every patient-facing
explanation produced by this layer. Wording is fixed; do not paraphrase
or generate variants.
"""

EXPLANATION_DISCLAIMER = (
    "This is an educational explanation based on validated clinical "
    "scoring criteria, not a diagnosis."
)


def disclaimer_html() -> str:
    """A minimal, visible HTML disclaimer component."""
    return (
        '<div class="explanation-disclaimer" role="note" '
        'style="border:1px solid #b7791f;background:#fffbe6;color:#744210;'
        'padding:10px 14px;border-radius:8px;font-size:14px;margin:12px 0;">'
        f"\u26a0\ufe0f {EXPLANATION_DISCLAIMER}"
        "</div>"
    )
