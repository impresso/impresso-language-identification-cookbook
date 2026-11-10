# Test run for language identification of image captions on BNL newspapers.
# Same options as v2-0-3, but writes to the sandbox buckets under its own run
# version so that production results stay untouched.
#
# Usage:
#   make collection \
#     CFG=configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-4-captions-test.mk

NEWSPAPER_FNMATCH ?= BNL/*

S3_BUCKET_CANONICAL ?= 112-canonical-final
S3_BUCKET_LANGIDENT_STAGE1 ?= 130-component-sandbox
S3_BUCKET_LANGIDENT ?= 140-processed-data-sandbox

RUN_VERSION_LANGIDENT ?= v2-0-4-captions-test

LANGIDENT_IMAGE_CAPTIONS_OPTION ?= --image-captions

include configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-3.mk
