# General NER with keras
## 模型结构 BiLSTM + CRF
![model](model.png)
## 训练结果
![res](res.png)
## 测试效果
Eg: '国务院总理在外交部长陈毅陪同下，访问了埃塞俄比亚'  
Re: ['person: 陈 毅 埃 塞', 'location: 俄 亚', 'organzation: 国']