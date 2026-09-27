from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_and_submits_a_sample_prediction():
    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(script).run()

    assert not app.exception
    assert app.title[0].value == "Heart Disease Risk Screening"

    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].label == "Model-estimated positive-class probability"
    assert app.error or app.success
