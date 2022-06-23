# 代码说明

## 环境配置

- Python 版本: 3.8.1 PyTorch 版本: 1.9.0 CUDA 版本: 11.1

- 所需环境在requirements.txt中定义

## 数据

- 仅使用大赛提供的未标注数据(100万)用于预训练

- 仅使用大赛提供的有标注数据(10万)用于训练及验证

- 未使用任何额外数据

## 预训练模型

- 使用了huggingface上提供的hfl/chinese-roberta-wwm-ext模型。链接为：[hfl/chinese-roberta-wwm-ext · Hugging Face](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

## 算法描述

- 本算法分单双流模型进行融合
- 对于双流模型，基于ALBEF算法的改进，文本编码器和多模态编码器使用ALBEF的6层Transformer。文本编码器使用BERT base模型的前6层进行初始化，多模态编码器使用BERT Base模型的最后6层进行初始化。文本编码器将输入文本转换为嵌入序列并输入多模态编码器；视觉特征在ALBEF处不做处理，送入多模态编码器与文本特征融合；取多模态编码器的各类输出并做最大池化和平均池化以及层归一化处理，同时将最初的视觉特征在此处使用nextvlad技术与各类输出直接拼接，最后过一次层归一化后通过简单的MLP结构去预测二级分类的id.
- 对于单流模型

## 性能

训练策略采用全量数据，未做验证。线上B榜测试性能：0.687352

## 训练流程

- 预训练部分
- 训练部分直接在有标注数据上训练，采用ema、pgd、fgm以及swa策略提高模型泛化性能

## 测试流程

- 双流模型训练4个epoch，取第2、3和4个epoch进行swa融合为一个模型
- 单流模型分为fgm和pgd策略，各训练4个epoch，pgd取第2、3个epoch进行swa融合，fgm取第2、3和4个epoch进行swa融合
- 融合6个模型，分别为：双流swa融合模型、双流训练第二个epoch模型、单流fgm的swa融合模型、单流fgm的第2个epoch模型、单流pgd的swa融合模型，单流pgd的第二个epoch模型。将融合后的模型来做测试。
