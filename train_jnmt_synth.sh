#!/bin/bash

SRC=$1
TGT=$2
CONFIG=$3

TIMESTAMP=`date +%s`
OUTPUT_DIR="../experiments/jnmt_synth_${CONFIG}_${SRC}-${TGT}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

HPARAMS="./nmt/standard_hparams/jnmt/${CONFIG}.json"
if [ ! -e ${HPARAMS} ]
then
  echo "${HPARAMS} not found."
  exit 1
fi

if [[ $CONFIG == c* ]]
then
  echo "Continuous model, using pre-trained word embeddings."
  EMBED="../data/${SRC}-${TGT}/word_vectors"
else
  echo "Discrete model, not using pre-trained word embeddings."
  EMBED="None"
fi

python -m nmt.nmt \
    --src=${SRC} \
    --tgt=${TGT} \
    --out_dir=${OUTPUT_DIR} \
    --vocab_prefix=../data/${SRC}-${TGT}/vocab \
    --train_prefix=../data/${SRC}-${TGT}/training \
    --dev_prefix=../data/${SRC}-${TGT}/dev \
    --synthetic_prefix=../data/${SRC}-${TGT}/training.mono_subsample \
    --embed_prefix=${EMBED} \
    --hparams_path=${HPARAMS} \
    &> ${OUTPUT_DIR}/log &

echo "Using config: ${HPARAMS}"
echo "You can check the logfile using:"
echo "less ${OUTPUT_DIR}/log"

if [[ -z $1 ]]; then TBPORT=6006; else TBPORT=$1; fi
echo "You can start tensorboard using:"
echo "tensorboard --logdir ${OUTPUT_DIR} --port ${TBPORT}"
