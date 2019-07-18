# Setting up environment

```
export PROJECT_DIR=$PWD
scl enable devtoolset-7 bash


wget https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.gz

tar -xzf boost_1_70_0.tar.gz
rm boost_1_70_0.tar.gz

cd boost_1_70_0/
./bootstrap.sh 
./b2 headers


cd $PROJECT_DIR

git clone https://github.com/psychocoderHPC/cupla.git
cd cupla
git merge origin/topic-standaloneHeader
cd example/CUDASamples/matrixMul/src
nvcc -x cu -w --generate-code arch=compute_70,code=sm_70 -O2 -std=c++14 --expt-relaxed-constexpr -include "cupla/standalone/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 -DALPAKA_DEBUG=0 -I$PROJECT_DIR/cupla/alpaka/include -I$PROJECT_DIR/cupla//include -I /usr/local/cuda-10.1/samples/common/inc/ -I$PROJECT_DIR/boost_1_70_0 matrixMul.cpp -o matrixMul

./matrixMul
```
