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



















































































# deploly_loongarch

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
