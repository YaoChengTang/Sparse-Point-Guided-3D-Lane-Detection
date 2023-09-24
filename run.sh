CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=21300 --nproc_per_node 1 main_implicitpersformer.py --mod=multiScaleRegCatHie_high_openlane --batch_size=8 --nepochs=100

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=20000 --nproc_per_node 1 main_persformer.py --mod=PersFormerAll --batch_size=8 --nepochs=100 --use_all_fea