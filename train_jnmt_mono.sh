#!/bin/bash

SRC=$1
TGT=$2
CONFIG=$3

TIMESTAMP=`date +%s`
OUTPUT_DIR="../experiments/jnmt_mono_${CONFIG}_${SRC}-${TGT}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

HPARAMS="./nmt/standard_hparams/jnmt/${CONFIG}.json"
if [ ! -e ${HPARAMS} ]
then
  echo "${HPARAMS} not found."
  exit 1
fi

python -m nmt.nmt \
    --src=${SRC} \
    --tgt=${TGT} \
    --out_dir=${OUTPUT_DIR} \
    --vocab_prefix=../data/${SRC}-${TGT}/vocab \
    --train_prefix=../data/${SRC}-${TGT}/training \
    --dev_prefix=../data/${SRC}-${TGT}/dev \
    --mono_prefix=../data/${SRC}-${TGT}/mono \
    --hparams_path=${HPARAMS} \
    &> ${OUTPUT_DIR}/log &

echo "Using config: ${HPARAMS}"
echo "You can check the logfile using:"
echo "less ${OUTPUT_DIR}/log"

if [[ -z $1 ]]; then TBPORT=6006; else TBPORT=$1; fi
echo "You can start tensorboard using:"
echo "tensorboard --logdir ${OUTPUT_DIR} --port ${TBPORT}"
