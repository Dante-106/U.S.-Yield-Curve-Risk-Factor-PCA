import pandas as pd

from yield_curve_pca.reporting import format_table


class _FrameWithoutStyler(pd.DataFrame):
    @property
    def style(self):
        raise AttributeError("The '.style' accessor requires jinja2")


def test_format_table_returns_frame_when_optional_styler_is_unavailable():
    frame = _FrameWithoutStyler({"Value": [1.0]})
    assert format_table(frame, formats={"Value": "{:.2f}"}, caption="Test") is frame
