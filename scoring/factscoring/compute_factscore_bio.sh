# Set your parent directory as an environment variable FACTUNE_DIR.
if [[ -z "${FACT_TUNE_DIR}" ]]; then
  echo "please set FACT_TUNE_DIR environment variable"
  exit
fi

# Set the path to fsenv as an environment variable. By default it is $FACTUNE_DIR/fsenv.
if [[ -z "${FSENV_PATH}" ]]; then
  FSENV_PATH=${FACT_TUNE_DIR}/fsenv
  if [ ! -d "$FSENV_PATH" ]; then
    echo "FSENV_PATH does not exist: $FSENV_PATH"
    exit
  fi
fi

# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/scoring/factscoring" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/scoring/factscoring"
  exit
fi
# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/data/dataset_bio" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/dataset_bio"
  exit
fi
# test if fs_cache_base exists
if [[ -z "${FACTSCORE_CACHE_BASE}" ]]; then
  echo "please set FACTSCORE_CACHE_BASE environment variable"
  exit
fi

if [ ! -d "${FACTSCORE_CACHE_BASE}" ]; then
  echo "Directory does not exist: ${FACTSCORE_CACHE_BASE}"
  exit
fi

# test if openai_key exists
if [ ! -f "${FACT_TUNE_DIR}/openai_key.txt" ]; then
  echo "File does not exist: ${FACT_TUNE_DIR}/openai_key.txt"
  exit
fi

cd ${FACT_TUNE_DIR}/scoring/factscoring
source ${FSENV_PATH}/bin/activate

python run_factscoring.py \
    --start 0 \
    --end 1 \
    --split train \
    --DATA_DIR ${FACT_TUNE_DIR}/data/dataset_bio \
    --oai_key ${FACT_TUNE_DIR}/openai_key.txt \
    --fs_cache_base ${FACTSCORE_CACHE_BASE} \
    --fs_cache_folder bio_cache
