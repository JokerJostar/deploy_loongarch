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
