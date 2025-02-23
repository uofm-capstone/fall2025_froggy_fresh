import os
import logging
import warnings

# Suppress extra logs and warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)