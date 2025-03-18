

python main_jmpbp.py --dataset="yelp_ddrm" --lr=0.005 --decay=0.0001 --early_stop=1 --warmup=10 --alpha=0.25  --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=0 --model="lgn" --patience=10 --comment "yelpddrm_mf"
python main_jmpbp.py --dataset="book_ddrm" --lr=0.006 --decay=0.0005 --early_stop=1 --warmup=10 --alpha=0.25 --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=0 --model="lgn" --patience=10 --comment="bookddrm_mf"
python main_jmpbp.py --dataset="ml_ddrm" --lr=0.00005 --decay=0.15 --early_stop=1 --warmup=5 --alpha=1.0 --topks=[10,20] --bpr_batch=256 --recdim=64 --layer=0 --model="lgn" --patience=5 --comment "mlddrm_mf"   

python main_jmpbp.py --dataset="yelp_ddrm" --lr=0.005 --decay=0.0001 --early_stop=1 --warmup=10 --alpha=0.125  --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=3 --model="lgn" --patience=10 --comment "yelpddrm"
python main_jmpbp.py --dataset="book_ddrm" --lr=0.005 --decay=0.0001 --early_stop=1 --warmup=10 --alpha=0.125  --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=3 --model="lgn" --patience=10 --gpu=1 --comment "bookddrm"
python main_jmpbp.py --dataset="ml_ddrm" --lr=0.0005 --decay=0.0001 --early_stop=1 --warmup=30 --alpha=0.25  --topks=[10,20] --bpr_batch=256 --recdim=64 --layer=3 --model="lgn" --patience=10 --gpu=1 --comment "mlddrm"

python main_jmpbp.py --dataset="yelp_ddrm" --lr=0.005 --decay=0.0001 --early_stop=1 --warmup=5 --alpha=0.5 --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=3 --model="sgl" --patience=5 --comment "yelpddrm_sgl"
python main_jmpbp.py --dataset="book_ddrm" --lr=0.005 --decay=0.0001 --early_stop=1 --warmup=5 --alpha=0.125  --topks=[10,20] --bpr_batch=2048 --recdim=64 --layer=3 --model="sgl" --patience=3 --comment "bookddrm_sgl" --test_interval=5
python main_jmpbp.py --dataset="ml_ddrm" --lr=0.0005 --decay=0.0001 --early_stop=1 --warmup=30 --alpha=0.5  --topks=[10,20] --bpr_batch=256 --recdim=64 --layer=3 --model="sgl" --patience=5 --comment "mlddrm_sgl"

