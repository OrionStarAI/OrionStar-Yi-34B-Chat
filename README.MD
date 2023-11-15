<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./pics/orion_start.PNG)

<div align="center">
<h1>
  OrionStar-Yi-34B-Chat
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OrionStarAI" target="_blank">Hugging Face</a> 
</p>

# 目录

- [📖 模型介绍](#模型介绍)
- [📊 模型推理 🔥](#模型推理)
- [👥 示例输出](#示例输出)
- [🥇 企业介绍](#企业介绍)
- [📜 声明、协议](#声明协议)

# 模型介绍

- OrionStar-Yi-34B-Chat 是猎户星空基于零一万物开源的**Yi-34B**模型，使用 __15W+__ 的高质量语料训练而来微调大模型，旨在为大模型社区用户提供卓越的交互体验。

- Yi系列模型是由零一万物团队开源的大模型，在多个权威的中文、英文及通用领域 benchmark
  上取得不错的效果。今天我们推出的Orionstar-Yi-34B-Chat更进一步挖掘了Yi-34B的潜力。通过对大量高质量微调语料库的深度训练，Orionstar-Yi-34B-Chat在评估数据上表现出色，我们致力于将其打造成为ChatGPT领域中的杰出开源替代品！

- 我们微调的模型对学术研究完全开放，同时请大家遵守[协议](#协议)
  和 [Yi License](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)

发布模型和下载链接见下表：

| 模型大小 | 下载链接                                                                                 | 
|------|--------------------------------------------------------------------------------------|
| 34B  | 🤗 [OrionStar-Yi-34B-Chat](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat) | 

# 模型推理

推理所需的模型权重、源码、配置已发布在 Hugging Face，下载链接见本文档最开始的表格。我们在此示范多种推理方式。程序会自动从
Hugging Face 下载所需资源。

## Python 代码方式

### Chat 模型推理方法示范

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat")
messages = [{"role": "user", "content": "你好! 你叫什么名字!"}]
response = model.chat(tokenizer, messages, streaming=Flase)
print(response)

# 你好，我的名字叫聚言，很高兴见到你。
```

在上述两段代码中，模型加载指定 `device_map='auto'`
，会使用所有可用显卡。如需指定使用的设备，可以使用类似 `export CUDA_VISIBLE_DEVICES=0,1`（使用了0、1号显卡）的方式控制。

## 命令行工具方式

```shell                                                                                                                                                                                                
python cli_demo.py                                                                                                                                                                  
```

本命令行工具是为 Chat 场景设计，因此我们不支持使用该工具调用 Base 模型。

## 脚本直接推理

```shell                                                                                                                                                                                                
 python text_generation.py --model OrionStarAI/OrionStar-Yi-34B-Chat --tokenizer OrionStarAI/OrionStar-Yi-34B-Chat --prompt 你好,你叫什么名字                                                                                                                                                                              
```

# 示例输出

## 闲聊

`````                                                                                           
用户：你好,你叫什么名字
OrionStar-Yi-34B：你好，我的名字叫聚言，很高兴见到你。
用户：你有哪些功能
OrionStar-Yi-34B：我是一个多才多艺的AI助手，我可以回答你的问题、陪你聊天、帮你查询信息、提供生活建议等等。
用户：讲个笑话
OrionStar-Yi-34B：为什么香肠总是不开心？因为它没有朋友，只能被夹在面包里。
`````

## 数学

`````                                                                                                                                                                                                   
用户：鸡和兔在一个笼子里，共有26个头，68只脚，那么鸡有多少只，兔有多少只？
OrionStar-Yi-34B：设鸡的数量为x，兔的数量为y。                                                                                                                                                          
根据题意，可得出以下两个方程：                                                                                                                                                                          
x + y = 26  （共有26个头）                                                                                                                                                                              
2x + 4y = 68 （共有68只脚）                                                                                                                                                                             
将第一个方程变形，得到y = 26 - x                                                                                                                                                                        
将y的值代入第二个方程中，得到：                                                                                                                                                                         
2x + 4(26-x) = 68                                                                                                                                                                                       
解这个方程，得到x = 18                                                                                                                                                                                  
所以，鸡有18只，兔有26 - 18 = 8只。
`````

# 企业介绍

猎户星空是一家全球领先的服务机器人解决方案公司，成立于2016年9月。ORION
STAR致力于基于人工智能技术打造下一代革命性机器人，使人们能够摆脱重复的体力劳动，使人类的工作和生活更加智能和有趣，通过技术使社会和世界变得更加美好。

猎户星空拥有完全自主开发的全链条人工智能技术，如语音交互和视觉导航。它整合了产品开发能力和技术应用能力。基于Orion机械臂平台，它推出了ORION
STAR AI Robot Greeting、AI Robot Greeting Mini、Lucki、Coffee
Master等产品，并建立了Orion机器人的开放平台OrionOS。通过为 **真正有用的机器人而生** 的理念实践，它通过AI技术为更多人赋能。

# 声明、协议

## 声明

我们在此声明，我们的开发团队并未基于 OrionStar-Yi-34B-Chat 模型开发任何应用，无论是在
iOS、Android、网页或任何其他平台。我们强烈呼吁所有使用者，不要利用 OrionStar-Yi-34B-Chat 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将
OrionStar-Yi-34B-Chat 模型用于未经适当安全审查和备案的互联网服务。

我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。
我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用
OrionStar-Yi-34B-Chat 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

## 协议

社区使用 OrionStar-Yi-34B-Chat
模型需要遵循 [Apache 2.0](https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/LICENSE)
和[《Yi-34B 模型社区许可协议》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)

# 联系我们

![](./pics/wechat_group.jpg)
