# This script sets several environment variables that are used in the llm_factuality_tuning codebase.
# The defaults are provide below. Please replace ??? with your own path.

# Path to llm_factuality_tuning codebase 
export FACT_TUNE_DIR=`pwd`

# Paths to python environments 
export FSENV_PATH=${FACT_TUNE_DIR}/fsenv
export CSENV_PATH=${FACT_TUNE_DIR}/csenv

# Path to your machine's local directory (not NFS) for speed
export LOCAL_DIR=???

# Path to your factscore .cache directory (read https://github.com/kttian/FActScore for info)
export FACTSCORE_CACHE_BASE=${LOCAL_DIR}/fct_scr/.cache 

# Path to on your file system to a folder to cache OpenAI queries
export NFS_OPENAI_CACHE=???


if [[ -z "${FACT_TUNE_DIR}" ]]; then
  echo "please set FACT_TUNE_DIR environment variable"
fi

# Set the path to csenv as an environment variable. By default it is $FACTUNE_DIR/csenv.
if [[ -z "${FSENV_PATH}" ]]; then
  FSENV_PATH=${FACT_TUNE_DIR}/fsenv
  if [ ! -d "$FSENV_PATH" ]; then
    echo "FSENV_PATH does not exist: $FSENV_PATH"
  fi
fi

# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/scoring/factscoring" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/scoring/factscoring"
fi
# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/data/dataset_bio" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/dataset_bio"
fi
# test if fs_cache_base exists
if [[ -z "${FACTSCORE_CACHE_BASE}" ]]; then
  echo "please set FACTSCORE_CACHE_BASE environment variable"
fi

if [ ! -d "${FACTSCORE_CACHE_BASE}" ]; then
  echo "Directory does not exist: ${FACTSCORE_CACHE_BASE}"
fi

# test if openai_key exists
if [ ! -f "${FACT_TUNE_DIR}/openai_key.txt" ]; then
  echo "File does not exist: ${FACT_TUNE_DIR}/openai_key.txt"
fi
