"""FlashStudio pages package — each page is a sub-package."""

from flashstudio.pages.dashboard.page import render_dashboard  # noqa: F401
from flashstudio.pages.data.page import render_data_page  # noqa: F401
from flashstudio.pages.model.page import render_model_page  # noqa: F401
from flashstudio.pages.training.page import render_training_page  # noqa: F401
from flashstudio.pages.export.page import render_export_page  # noqa: F401
from flashstudio.pages.inference.page import render_inference_page  # noqa: F401
