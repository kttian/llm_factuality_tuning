# Set your parent directory as an environment variable FACTUNE_DIR.
if [[ -z "${FACT_TUNE_DIR}" ]]; then
  echo "please set FACT_TUNE_DIR environment variable"
  exit
fi

# Set the path to csenv as an environment variable. By default it is $FACTUNE_DIR/csenv.
if [[ -z "${CSENV_PATH}" ]]; then
  CSENV_PATH=${FACT_TUNE_DIR}/csenv
  # test if the directory exists
  if [ ! -d "$CSENV_PATH" ]; then
    echo "CSENV_PATH does not exist: $CSENV_PATH"
    exit
  fi
fi

# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/scoring/confidence_scoring" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/scoring/confidence_scoring"
  exit
fi
# test if the directory exists
if [ ! -d "${FACT_TUNE_DIR}/data/dataset_bio" ]; then
  echo "Directory does not exist: ${FACT_TUNE_DIR}/data/dataset_bio"
  exit
fi

cd ${FACT_TUNE_DIR}/scoring/confidence_scoring
source ${CSENV_PATH}/bin/activate
DATA_DIR=${FACT_TUNE_DIR}/data/dataset_bio

# Step 1: Convert atomic facts to questions
python atomic_fact_to_q.py --start 0 --end 1 \
    --DATA_DIR $DATA_DIR --dataset bio \
    --openai_model gpt-3.5-turbo-0613 \
    --split train 

# Step 2: Sample answers to atomic questions
python atomic_sample_answers.py --start 0 --end 1 \
    --DATA_DIR $DATA_DIR --dataset bio \
    --split train
    
# Step 3 (Optional): Compute entity-based confidence rewards rewards
python entity_sample.py --start 0 --end 1 \
    --DATA_DIR $DATA_DIR --dataset bio \
    --split train

# Step 4: Compute confidence-based rewards 
python score_metrics.py \
    --DATA_DIR $DATA_DIR --dataset bio \
    --split train 

# Repeat for train and dpoval splits.