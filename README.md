<h1>本仓库是对DAML论文的复现代码</h1>
1.main.py提供该项目的主入口，kwargs填写模型的各项需要进行设置的参数。具体参数参考config文件。<br>
2.设置完毕后，使用 **train(kwargs)** 函数，即可训练模型。模型将会自动保存到 dataset 文件夹中。<br>
3.使用 **test(kwargs)** 函数，可以进行模型的测试，模型测试的结果也将会自动保存到dataset 文件夹中。<br>
4.如果需要添加模型，需要在 model 文件夹中编写模型。编写的模型需要在**forward()** 中返回经过模型处理后的特征向量。