### 问题记录

* pycharm远程服务器报错`runnerw.exe: CreateProcess failed with error 2 `
  * 运行`jupyter lab list`或`jupyter notebook list`查看地址+token
  * 打开pycharm设置中的jupyter servers，把地址+token输入到configured server栏目中
* `a=df.groupby()`相当于一个view，每当df更新时，a的内容会自动更新
* python进程结束后，显存不释放：执行`sudo kill -9 'pgrep python'`杀掉python进程



### bug记录

- [x] 数据增量更新时，增量部分的股票代码和日期出现丢失
- [x] xgboost时序模型效果极差，原因排查
  - [x] 特征工程数据泄露：股票数据为面板数据，未对不同只股票数据分别进行操作，导致跨股票开展特征工程，异常值很多（30min）
  - [x] 绝对收益率与**相对超额收益率**的选取：一定要选择后者，否则市场上众多噪声会导致模型难以学习到合理的模式
  - [x] feature_data生成未来x日超额收益率时，会导致模型索引混乱
- [ ] agent调用函数不准确，无法正确获取用户参数：改用工程方法，手写参数获取函数，在函数内调用大模型实现多轮智能对话-2
  - [ ] 意图识别：
  - [ ] 记忆机制（用户历史输入的参数保存到记忆中，下次调用该函数时先从本地读取）
- [ ] 建模不够细致，不能针对不同的金融产品采用不同的模型-1
- [ ] 金融模型的表现效果较差，需要更换模型&重构特征-1
- [ ] 文本数据库的内容不够丰富，该如何增强-4
- [ ] reactagent的自有函数QueryEngineTool在处理rag时表现较好，需要将ML部分的自定义对话函数与RAG部分的reactagent函数结合起来-3