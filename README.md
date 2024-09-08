# A MoE Multimodal Graph Attention Networks Framework for Multimodal Emotion Recognition


## Requirements

- Python 3.8.12
- PyTorch 1.11.0+cu113



### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

The pre-extracted multimodal features and text semantic relationships are available at [IEMOCAP](https://www.dropbox.com/scl/fi/ir3bfddyczx3gcfszffim/IEMOCAP.zip?rlkey=lpahvoqdmsijad0kis38paqyw&st=u8svfp8l&dl=0) and [MELD](https://www.dropbox.com/scl/fi/045w04sven4t9nmqs4nm0/MELD.zip?rlkey=7negyyo1tqt7rp4kuh00w738y&st=hdrltlkj&dl=0), or you can manually extract semantic relationships by yourself at [DDP_Parsing](https://github.com/seq-to-mind/DDP_parsing).


### Training examples
For IEMOCAP
```python
python main.py --dataset IEMOCAP --lr 5e-5 --dropout 0.2 --num_experts 3 --seed 100 --batch_size 8 --gnn_layers 2
```
For MELD:
```python
python main.py --dataset MELD --lr 5e-5 --dropout 0.2 --num_experts 3 --seed 100 --batch_size 32 --gnn_layers 2
```


### Eval

For IEMOCAP
```python
python eval.py --batch_size 8 --dataset IEMOCAP
```
For MELD:
```python
python eval.py --dataset MELD
```
