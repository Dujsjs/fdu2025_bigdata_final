### 问题记录

* pycharm远程服务器报错`runnerw.exe: CreateProcess failed with error 2 `
  * 运行`jupyter lab list`或`jupyter notebook list`查看地址+token
  * 打开pycharm设置中的jupyter servers，把地址+token输入到configured server栏目中
* `a=df.groupby()`相当于一个view，每当df更新时，a的内容会自动更新



### bug记录

- [x] 数据增量更新时，增量部分的股票代码和日期出现丢失
- [ ] xgboost时序模型效果极差，原因排查
  - [x] 特征工程数据泄露：股票数据为面板数据，未对不同只股票数据分别进行操作，导致跨股票开展特征工程，异常值很多（30min）
  - [x] 绝对收益率与**相对超额收益率**的选取：一定要选择后者，否则市场上众多噪声会导致模型难以学习到合理的模式
  - [ ] feature_data生成未来x日超额收益率时，会导致模型索引混乱