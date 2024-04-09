<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./pics/orion_start.PNG)

<div align="center">
<h1>
  OrionStar-Yi-34B-Chat
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat" target="_blank">HuggingFace 模型下载</a> |  🤖 <a href="https://modelscope.cn/models/OrionStarAI/OrionStar-Yi-34B-Chat/summary" target="_blank">ModelScope 模型下载</a>|  🧰 <a href="https://openxlab.org.cn/models/detail/OrionStarAI/OrionStar-Yi-34B-Chat" target="_blank">OpenXLab 模型下载</a>
</p>


<p align="center">
🤗 <a href="https://modelscope.cn/studios/OrionStarAI/OrionStar-Yi-34B-Chat/summary" target="_blank">💡在线Demo</a> 
</p>


<div align="center">


<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/README_en.MD">English</a>
    <p>
</h4>

</div>

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

| 模型大小 | Hugging Face 下载链接                                                             | OpenXLab 下载链接                                                             |  
|------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 34B  | 🤗 [OrionStar-Yi-34B-Chat](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat) |  🧰 [OrionStar-Yi-34B-Chat](https://openxlab.org.cn/models/detail/OrionStarAI/OrionStar-Yi-34B-Chat) | 

- 模型评估结果

我们使用[opencompass](https://opencompass.org.cn)对以下通用领域数据集进行了 5-shot
测试。其他模型评估结果取自[opencompass-leaderboard](https://opencompass.org.cn/leaderboard-llm)。

|                           | C-Eval    | MMLU   | CMMLU     |
|---------------------------|-----------|--------|-----------|
| **GPT-4**                 | 69.9      | **83** | 71        |
| **ChatGPT**               | 52.5      | 69.1   | 53.9      | 			
| **Claude-1**              | 52        | 65.7   | -         |
| **TigerBot-70B-Chat-V2**  | 57.7      | 65.9   | 59.9      |
| **WeMix-LLaMA2-70B**      | 55.2      | 71.3   | 56        |  			
| **LLaMA-2-70B-Chat**      | 44.3      | 63.8   | 43.3      |
| **Qwen-14B-Chat**         | 71.7      | 66.4   | 70        |
| **Baichuan2-13B-Chat**    | 56.7      | 57     | 58.4      |      	
| **OrionStar-Yi-34B-Chat** | **77.71** | 78.32  | **73.52** |  

# 模型推理

推理所需的模型权重、源码、配置已发布在 Hugging Face，下载链接见本文档最开始的表格。我们在此示范多种推理方式。程序会自动从
Hugging Face 下载所需资源。

## Python 代码方式

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

猎户星空（ORION
STAR）是一家全球领先的服务机器人解决方案公司，成立于2016年9月。猎户星空致力于基于人工智能技术打造下一代革命性机器人，使人们能够摆脱重复的体力劳动，使人类的工作和生活更加智能和有趣，通过技术使社会和世界变得更加美好。

猎户星空拥有完全自主开发的全链条人工智能技术，如语音交互和视觉导航。它整合了产品开发能力和技术应用能力。基于Orion机械臂平台，它推出了ORION
STAR AI Robot Greeting、AI Robot Greeting Mini、Lucki、Coffee
Master等产品，并建立了Orion机器人的开放平台OrionOS。通过为 **真正有用的机器人而生** 的理念实践，它通过AI技术为更多人赋能。

凭借7年AI经验积累，猎户星空已推出的大模型深度应用“聚言”，并陆续面向行业客户提供定制化AI大模型咨询与服务解决方案，真正帮助客户实现企业经营效率领先同行目标。

**猎户星空具备全链条大模型应用能力的核心优势**，包括拥有从海量数据处理、大模型预训练、二次预训练、微调(Fine-tune)、Prompt
Engineering 、Agent开发的全链条能力和经验积累；拥有完整的端到端模型训练能力，包括系统化的数据处理流程和数百张GPU的并行模型训练能力，现已在大政务、云服务、出海电商、快消等多个行业场景落地。

***欢迎有大模型应用落地需求的企业联系我们进行商务合作，咨询电话 400-898-7779 。***

企业微信

![](./pics/company_wechat.jpg)

# 声明、协议

## 声明

我们强烈呼吁所有使用者，不要利用 OrionStar-Yi-34B-Chat 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将
OrionStar-Yi-34B-Chat 模型用于未经适当安全审查和备案的互联网服务。

我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。
我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用
OrionStar-Yi-34B-Chat 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

## 协议

社区使用 OrionStar-Yi-34B-Chat
模型需要遵循 [Apache 2.0](https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/LICENSE)
和[《Yi-34B 模型社区许可协议》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)

# 联系我们
***欢迎有大模型应用落地需求的企业联系我们进行商务合作***<br>
**咨询电话:** 400-898-7779<br>
**电子邮箱:** ai@orionstar.com<br>
**Discord社区链接:** https://discord.gg/zumjDWgdAs

<div align="center">
  <img src="./pics/wechat_group.jpg" alt="wechat" width="40%" />
</div>
