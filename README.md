# 环境安装

`conda env create -f environment.yml`

上面的命令不会安装paddlepaddle-gpu，需要手动用pip再安装一下，参考[这个地址](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，注意，用pip安装包含cuDNN动态链接库的paddlepaddle。


```
python3 -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
python3 -m pip install paddleocr
```

安装mmcv，mmcv提供了很多基础的视频/图片操作

```
pip install openmim
mim install mmcv
```

安装TensorRT
```
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
```
如果开启TensorRT报错，显示无法找到动态库，按照下面的方法设置软链接，让PaddleOCR找到动态库：
```
(ppocr) dtong@QX-AI-10:~/miniconda3/envs/ppocr/lib/python3.8/site-packages/tensorrt_libs$ ln -s libnvinfer.so.8 libnvinfer.so
(ppocr) dtong@QX-AI-10:~/miniconda3/envs/ppocr/lib/python3.8/site-packages/tensorrt_libs$ ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so
(ppocr) dtong@QX-AI-10:~/miniconda3/envs/ppocr/lib/python3.8/site-packages/tensorrt_libs$ ln -s libnvinfer_builder_resource.so.8.6.1 libnvinfer_builder_resource.so
```
更新了算法说明及测试结果，在对应的ocr文件夹里，可根据需求及目标进行选择
```
权重链接：
链接：https://pan.baidu.com/s/1qtJynXuxwcC2fmK4HLMqgQ?pwd=inve 
提取码：inve
### trt构建
通过trt构建流程PDF可以进行paddle-ocr的trt推理，然后将trt推理框架加入主函数实现ocr的加速检测及识别，其中有对应的脚本文件及注意事项，方便小白使用

### 结果
可在目录下找到

