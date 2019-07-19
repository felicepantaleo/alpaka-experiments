/**
   Vector addition:
   takes vectors a and b as input, computes vector sum 
   and stores output in vector c

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 07/2019

 */

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <typeinfo>

#include <alpaka/alpaka.hpp>

using namespace std;


struct vector_addition_kernel{
  ALPAKA_NO_HOST_ACC_WARNING
  template<
    typename TAcc,
    typename TElem,
    typename TIdx>
  ALPAKA_FN_ACC auto operator()(
                                TAcc const & acc,
                                TElem const * const a,
                                TElem const * const b,
                                TElem * const c,
                                TIdx const & vec_size) const
    -> void
  {
    TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
    TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
    TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);
    
    if(threadFirstElemIdx < vec_size)
      {
        TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
        TIdx const threadLastElemIdxClipped((vec_size > threadLastElemIdx) ? threadLastElemIdx : vec_size);
        
        for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
          {
            c[i] = c[i] + b[i];
          }
      }
  }
};

int main(int argc, char *argv[] ) {

  if ( argc != 4 ) {
    cout << "Need two arguments: size of vector, number of threads / block and device to use" << endl;
    return -1;
  }
  
  const int vec_size_h  = atoi(argv[argc-3]);
  const int n_threads = atoi(argv[argc-2]);
  const int device_id = atoi(argv[argc-1]);

  /* Chose device to use */
  //cudaSetDevice(device_id);
  
  cout << "Adding vectors of size " <<  vec_size_h << " with " << n_threads << " threads" << endl;

  /* Alpaka initialization */ 
  // Configure Alpaka
  using Dim     = alpaka::dim::DimInt<1u>;
  using Size    = std::size_t;
  using Acc     = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Host    = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Stream  = alpaka::stream::StreamCpuSync;
  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
  using Elem    = int;

  // Retrieve devices and stream
  using DevHost = alpaka::dev::DevCpu;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  DevHost devHost (alpaka::pltf::getDevByIdx<PltfHost>(0u));
  DevAcc  devAcc  (alpaka::pltf::getDevByIdx<PltfAcc>(0) );
  Stream  stream  ( devAcc);

  // Specify work division
  const int n_blocks  = vec_size_h / n_threads + (vec_size_h % n_threads != 0);
  auto elementsPerThread ( alpaka::vec::Vec<Dim, Size>::ones() );
  auto threadsPerBlock   ( alpaka::vec::Vec<Dim, Size>::all(n_threads) );
  alpaka::vec::Vec<Dim, Size> const blocksPerGrid( static_cast<Size>(n_blocks) );
  WorkDiv workdiv(
    alpaka::workdiv::WorkDivMembers<Dim, Size>(
      blocksPerGrid,
      threadsPerBlock,
      elementsPerThread)
  );
  
  /* Host memory for the two input vectors a and b and the output vector c */
  Size const numElements(vec_size_h);
  alpaka::vec::Vec<Dim, Size> const extent(numElements);
  using BufHost = alpaka::mem::buf::Buf<DevHost, int, Dim, Size>;
  using Data = int;
  BufHost a_h = alpaka::mem::buf::alloc<Data, Size>(devHost, extent);
  BufHost b_h = alpaka::mem::buf::alloc<Data, Size>(devHost, extent);
  BufHost c_h = alpaka::mem::buf::alloc<Data, Size>(devHost, extent);

  const auto a_h_native = alpaka::mem::view::getPtrNative(a_h);
  
  for ( Size i = 0; i < numElements; i++ ) {
    alpaka::mem::view::getPtrNative(a_h)[i] = i;
    alpaka::mem::view::getPtrNative(b_h)[i] = i;
    alpaka::mem::view::getPtrNative(c_h)[i] = 0;
  }
  
  /* Device pointers for the three vectors a, b, c */
  using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Size>;
  BufAcc a_acc = alpaka::mem::buf::alloc<Data, Size>(devAcc, extent);
  BufAcc b_acc = alpaka::mem::buf::alloc<Data, Size>(devAcc, extent);
  BufAcc c_acc = alpaka::mem::buf::alloc<Data, Size>(devAcc, extent);

  
  /* Copy vectors to accelerator */
  alpaka::mem::view::copy(stream, a_acc, a_h, extent);
  alpaka::mem::view::copy(stream, b_acc, b_h, extent);

  /* Instantiate the kernel function object */
  vector_addition_kernel kernel;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  /* Create kernel execution task */
  auto const taskKernel(alpaka::exec::create<Acc>(
   workdiv,
   kernel,
   alpaka::mem::view::getPtrNative(a_acc),
   alpaka::mem::view::getPtrNative(b_acc),
   alpaka::mem::view::getPtrNative(c_acc),
   numElements));

  /* Enqueue the kernel execution task */
  alpaka::stream::enqueue(stream, taskKernel);

  
  /* Copy back the result */
  alpaka::mem::view::copy(stream, c_h, c_acc, extent);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  for ( int i = 0; i < vec_size_h; i++ ) {
    cout << alpaka::mem::view::getPtrNative(a_h)[i] << " + " << alpaka::mem::view::getPtrNative(b_h)[i] << " = " << alpaka::mem::view::getPtrNative(c_h)[i] << endl;
  }

  cout << "Kernel duration: " << elapsed_seconds.count() << " s " << endl;
  cout << "Time per kenel: " << elapsed_seconds.count() / vec_size_h << endl;
    
  return 0;
}
