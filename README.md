# DenoisingRec
# main_coteaching直接运行即可，就是最好的版本Recall 0.0994-0.2345；NDCG 0.0853-0.1335

Adaptive Denoising Training for Recommendation.

This is the pytorch implementation of our paper at WSDM 2021:

> [Denoising Implicit Feedback for Recommendation.](https://arxiv.org/abs/2006.04153)<br>
> Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua.

## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

For others, please refer to the file env.yaml.

## Usage

### Training
#### T_CE
```
python main.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```
or use run.sh
```
sh run.sh dataset model drop_rate num_gradual gpu_id
```
The output will be in the ./log/xxx folder.

#### R_CE
```
sh run.sh dataset model alpha gpu_id
```
### Inference
We provide the code to inference based on the well-trained model parameters.
```
python inference.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5
```
### Examples
1. Train GMF by T_CE on Yelp:
```
python main.py --dataset=yelp --model=GMF --drop_rate=0.1 --num_gradual=30000 --gpu=0
```
2. Train NeuMF by R_CE on Amazon_book
```
python main.py --dataset=amazon_book --model=NeuMF-end --alpha=_0.25 --gpu=0
```
We release all training logs in ./log folder. The hyperparameter settings can be found in the log file. 
The well-trained parameter files are too big to upload to Github. I will upload to drives later and share it here.

## Citation  
If you use our code, please kindly cite:

```
@inproceedings{wang2021denoising,
  title={Denoising implicit feedback for recommendation},
  author={Wang, Wenjie and Feng, Fuli and He, Xiangnan and Nie, Liqiang and Chua, Tat-Seng},
  booktitle={Proceedings of the 14th ACM international conference on web search and data mining},
  pages={373--381},
  publisher={ACM},
  year={2021}
}
```
## Acknowledgment

Thanks to the NCF implementation:
- [Tensorflow Version](https://github.com/hexiangnan/neural_collaborative_filtering) from Xiangnan He. 
- [Torch Version](https://github.com/guoyang9/NCF) from Yangyang Guo.

Besides, this research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative, and the National Natural Science Foundation of China (61972372, U19A2079). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore. 

## License

NUS © [NExT++](https://nextcenter.org/)


# 2023/5/4中午
13:07: main_coteaching.py的效果不好，只是比单模型版本多了一个模型，但也是分开训练的，效果会差。
现在加一个main_Coteaching，再在单模型的基础上重改试一试。
两个模型在一起，但是不合并loss，效果与原先相当
接着直接用co-teaching的loss看效果