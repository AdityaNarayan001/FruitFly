#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

static thread_local std::string last_error;
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static void alloc(T*& p,size_t n) { check(cudaMalloc((void**)&p, (n?n:1)*sizeof(T))); }
struct Model {
  int n,delay,refractory; long long tick=0; float ratio,decay,rest,threshold;
  int64_t* ptr=nullptr; int32_t* idx=nullptr; float *weights=nullptr,*v=nullptr,*s=nullptr,*ring=nullptr,*z=nullptr;
  int *ref=nullptr,*counts=nullptr;
  ~Model(){ cudaFree(ptr);cudaFree(idx);cudaFree(weights);cudaFree(v);cudaFree(s);cudaFree(ring);cudaFree(z);cudaFree(ref);cudaFree(counts); }
};
__global__ void reset_kernel(int n,float rest,float* v,float* s,int* ref) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){v[i]=rest;s[i]=0;ref[i]=0;}
}
__global__ void step_kernel(int n,const int64_t* ptr,const int32_t* idx,const float* w,
  float* v,float* s,int* ref,const float* delayed,float* z,const float* drive,int* count,
  float decay,float ratio,float rest,float threshold,int refractory) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float incoming=0;
  for(int64_t e=ptr[i];e<ptr[i+1];++e) incoming+=w[e]*delayed[idx[e]];
  s[i]=decay*s[i]+incoming; z[i]=0;
  if(ref[i]>0){ --ref[i]; return; }
  v[i]+=ratio*(rest-v[i])+s[i]+drive[i];
  if(v[i]>=threshold){z[i]=1;v[i]=rest;ref[i]=refractory;++count[i];}
}
extern "C" const char* ffb_error(){return last_error.c_str();}
extern "C" int ffb_reset(void* handle) {
  try {auto* m=(Model*)handle; m->tick=0;
    reset_kernel<<<(m->n+255)/256,256>>>(m->n,m->rest,m->v,m->s,m->ref);
    check(cudaMemset(m->ring,0,(size_t)m->n*m->delay*sizeof(float)));check(cudaDeviceSynchronize());return 0;
  }catch(const std::exception& e){last_error=e.what();return -1;}
}
extern "C" void* ffb_create(int n,int64_t edges,const int64_t* ptr,const int32_t* idx,const float* weights,
  float ratio,float decay,float rest,float threshold,int delay,int refractory) {
  Model* m=nullptr;
  try {m=new Model();m->n=n;m->ratio=ratio;m->decay=decay;m->rest=rest;m->threshold=threshold;m->delay=delay;m->refractory=refractory;
    alloc(m->ptr,n+1);alloc(m->idx,edges);alloc(m->weights,edges);alloc(m->v,n);alloc(m->s,n);
    alloc(m->ring,(size_t)n*delay);alloc(m->z,n);alloc(m->ref,n);alloc(m->counts,n);
    check(cudaMemcpy(m->ptr,ptr,((size_t)n+1)*sizeof(int64_t),cudaMemcpyHostToDevice));
    if(edges){check(cudaMemcpy(m->idx,idx,edges*sizeof(int32_t),cudaMemcpyHostToDevice));check(cudaMemcpy(m->weights,weights,edges*sizeof(float),cudaMemcpyHostToDevice));}
    if(ffb_reset(m))throw std::runtime_error(last_error);return m;
  }catch(const std::exception& e){last_error=e.what();delete m;return nullptr;}
}
extern "C" int ffb_run(void* handle,int steps,const float* drive,int* counts,float* voltage,float* spikes,float* volts) {
  float *input=nullptr,*zs=nullptr,*vs=nullptr;
  try {auto* m=(Model*)handle;size_t cells=(size_t)steps*m->n, bytes=cells*sizeof(float);
    alloc(input,cells);check(cudaMemcpy(input,drive,bytes,cudaMemcpyHostToDevice));
    if(spikes)alloc(zs,cells);if(volts)alloc(vs,cells);check(cudaMemset(m->counts,0,m->n*sizeof(int)));
    for(int t=0;t<steps;++t){
      float* slot=m->ring+(m->tick%m->delay)*m->n;
      step_kernel<<<(m->n+255)/256,256>>>(m->n,m->ptr,m->idx,m->weights,m->v,m->s,m->ref,slot,m->z,input+(size_t)t*m->n,m->counts,m->decay,m->ratio,m->rest,m->threshold,m->refractory);
      check(cudaGetLastError());
      // Commit only after every neuron has read the old delayed spike vector.
      check(cudaMemcpy(slot,m->z,m->n*sizeof(float),cudaMemcpyDeviceToDevice));
      if(zs)check(cudaMemcpy(zs+(size_t)t*m->n,m->z,m->n*sizeof(float),cudaMemcpyDeviceToDevice));
      if(vs)check(cudaMemcpy(vs+(size_t)t*m->n,m->v,m->n*sizeof(float),cudaMemcpyDeviceToDevice));
      ++m->tick;
    }
    check(cudaDeviceSynchronize());check(cudaMemcpy(counts,m->counts,m->n*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(voltage,m->v,m->n*sizeof(float),cudaMemcpyDeviceToHost));
    if(spikes)check(cudaMemcpy(spikes,zs,bytes,cudaMemcpyDeviceToHost));
    if(volts)check(cudaMemcpy(volts,vs,bytes,cudaMemcpyDeviceToHost));
    cudaFree(input);cudaFree(zs);cudaFree(vs);return 0;
  }catch(const std::exception& e){last_error=e.what();cudaFree(input);cudaFree(zs);cudaFree(vs);return -1;}
}
extern "C" void ffb_destroy(void* handle){delete (Model*)handle;}
