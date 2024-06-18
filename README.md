# 概览

个人想法是loongarch64平台不止可以作为单片机部署，可以有更广泛的应用场景，比如loongarch平台的个人电脑，或者基于loongarch的算力集群（可能现在还没有），于是该项目以IESD-2024大赛为契机，本人自2024年5月19日开始搭建该项目，旨在通过构建一个完整的AIoT框架，实现从训练到部署的全链路，以便在龙芯平台上实现应用的部署甚至是训练。

训练部分基于本人开发的ATML无监督训练框架

关于模型结构与训练框架请前往[ECM-ATML_loongarch64](https://github.com/JokerJostar/ECM-ATML_loongarch64)

本库主要阐述在部署方面的工作


# pytorch路线

采用此路线的原因是想通过构建运行时来实现  训练-部署 全链路pytorch框架


采用以下两种方案实现

executorch

libtorch

## excutorch

使用executorch库构建运行时，对交叉编译工具兼容性好

#### 1. 克隆并编译安装executorch

#### 2. 将本项目4executorch下的文件分别移动到executorch目录下的指定位置

``executor_runner.cpp -> excutorch/examples/portable/executor_runner/executor_runner.cpp``

``inputs_portable.cpp -> excutorch/extension/runner_util/inputs_portable.cpp``

``inputs_portable.h -> excutorch/extension/runner_util/inputs_portable.h``

``loongarch64-toolchain.cmake -> excutorch/loongarch64-toolchain.cmake``

#### 3. 使用交叉编译工具链编译生成executor_runner可执行文件


#### 4.使用2pte.py转化模型

#### 5.上板部署运行，与eveluation_af_detection.py通信

### 优点

1.支持最新算子

2.支持fp32

### 缺点

由于时间问题，还未解决量化问题，不过可以通过xnnpack解决


## libtorch

libtorch的项目搭建简单，但是所有的libtorch链接库需要重新编译，对交叉编译工具链的兼容性不好
#### 1.将lib导入环境变量

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lib```

#### 2.使用交叉编译工具链进行编译生成可执行文件

### 优点

因为是加载的pytorch训练后原生script模型文件，不用经过转换，理论上算子支持是最完整的，而且不论是否量化都可以推理

### 缺点

上板后bus erro，由于生成构建libtorch时没有强制内存对齐，而loongarch64应该是不支持非对齐访问的，所以需要重新编译libtorch，但是这个工程量太大，需要官方提供支持


# tfLite路线

采用此路线的原因是tfLite的模型转换工具支持量化，可以在模型转换时直接量化,但是算子兼容非常有限

### tflite间接训练部署

#### 1.使用ATML框架训练模型后导出为onnx模型

#### 2.使用onnx2tf转换为pd模型

#### 3.使用tfquant.py脚本转换为量化为int8的tflite模型

#### 4.基于官方例程使用tflite-micro进行编译部署


### tflite直接训练部署

#### 直接用tftrain.py训练，不需要cvs索引加载数据，也不用任何转换直接保存为tflite格式，并且可以直接编译上板（时间原因未合并进ATML主线）

### 优点

模型文件最小可以达到9kb，可执行文件可以达到140kb左右，推理速度可达5ms

### 缺点

算子支持少，精度支持不丰富



















































































# 备注

```
export PATH=$PATH:/home/jostar/workspace/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3-1/bin/
export CROSS_COMPILE=loongarch64-linux-gnu
export ARCH=loongarch64
```
```
rm -rf cmake-out && mkdir cmake-out && cd cmake-out
```


```
cmake -DCMAKE_TOOLCHAIN_FILE=../loongarch64-toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-Os -ffunction-sections -fdata-sections" -DCMAKE_CXX_FLAGS="-Os -ffunction-sections -fdata-sections"  ..
```
