import importlib
import sys

# Metaflow only auto-packages modules that define a FlowMutator/user_step_decorator
# (like `common.pipeline`); everything else must opt in explicitly. The `register`
# step in the Training pipeline needs the actual `inference/backend.py` and
# `inference/model.py` files on disk (via `mlflow.pyfunc.log_model`), so we need
# this package shipped to remote compute platforms too.
METAFLOW_PACKAGE_POLICY = "include"

# We want to register the submodules of the inference package as top-level modules
# so we can import them directly from the inference package without having to
# specify their "inference" prefix.
submodules = ["backend"]
for submodule in submodules:
    module_name = f"{__name__}.{submodule}"
    module = importlib.import_module(module_name)
    sys.modules[submodule] = module
