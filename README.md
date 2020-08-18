# aes

英文作文自动评分



 安装：
在python3.7下安装torch，numpy，collections, pandas, math, nltk, sklearn包后，通过pycharm直接打开运行。

因glove文件太大无法上传，需自行下载




基于lstm，使用glove作为词向量训练，并加入每篇文章的句子个数、单词总数、单词平均长度、拼写错误数作为训练参数，使用LSTM在pytorch下进行训练与预测，最后对验证集通过MSE，RMSE，Cohen's k, Pearson r, Spearman's ρ对预测结果进行评分。
