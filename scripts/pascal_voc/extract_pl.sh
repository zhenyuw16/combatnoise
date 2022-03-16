
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

bash tools/dist_test.sh configs/combatnoise/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_generatepl.py  work_dirs/faster_rcnn_r50_fpn_1x_voc07_sup/epoch_4.pth $GPUS --out $RESULTNAME

python scripts/pascal_voc/pkl2json.py $RESULTNAME

python scripts/pascal_voc/filter_pl.py $RESULTNAME

python scripts/pascal_voc/form_ann.py $RESULTNAME $ANNFILE
