# relation_extraction_hw

模型引用自[RIFRE](https://github.com/zhao9797/RIFRE)

## 准备

1. clone到作业目录下，使用`preprocess.py`将数据进行预处理
2. 从[google drive](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK)下载`bert-base-uncased`和`bert-base-cased`，放在`RIFRE-main\datasets\bert`目录下

## 训练

```python
cd RIFRE-main/
python train.py
```

## 测试

```python
cd RIFRE-main/
python test.py
```

测试结果放在`RIFRE-main\datasets\data\hw\test_result.txt`中