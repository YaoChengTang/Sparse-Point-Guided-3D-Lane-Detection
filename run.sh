python -m torch.distributed.launch --nproc_per_node 1 main_persformer.py --mod=OpenLaneVis --batch_size=3 --vis_seg_id="7999729608823422351_1483_600_1503_600"

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=21300 --nproc_per_node 1 main_implicitpersformer_high.py --mod=multiScaleRegCatHie_high_openlane --batch_size=8 --nepochs=100

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=20000 --nproc_per_node 1 main_persformer.py --mod=PersFormerAll --batch_size=8 --nepochs=100 --use_all_fea