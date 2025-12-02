OUTPUT_DIR = "./results"

# Language configuration for NLLB-200
SRC_LANG = "zho_Hans"
TGT_LANG = "eng_Latn"

# Sequence length settings tuned for sentence-level translation
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

# Base model selection
MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"

# Training sampling parameters
TRAIN_SAMPLES = 500_000
VAL_SAMPLES = 2_000
