python main_Coteaching_lightgcn_BCE.py --dataset=yelp --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]
python main_Coteaching_lightgcn_TCE.py --dataset=yelp --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]
python main_Coteaching_lightgcn_Co.py --dataset=yelp --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]

python main_Coteaching_lightgcn_BCE.py --dataset=amazon_book --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]
python main_Coteaching_lightgcn_TCE.py --dataset=amazon_book --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]
python main_Coteaching_lightgcn_Co.py --dataset=amazon_book --gpu=3 --num_gradual=30000 --drop_rate=0.2 --top_k=[50,100]



