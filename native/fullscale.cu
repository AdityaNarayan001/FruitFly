// Independent bounded recurrent model. No changes to the legacy LIF backend.
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>
static thread_local std::string error;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
struct Net {
 int n,m; int *ptr,*idx,*post,*tp,*order,*channel; float *base,*w,*sign,*r,*b;
 float *h0,*h1,*h2,*t1,*t2,*p0,*p1,*p2,*z,*gz1,*gz2,*back,*obs,*q,*qp,*norm,*delta;
 unsigned char *seen; bool cache=false; std::vector<void*> allocations;
 template<class T> void alloc(T*& p,size_t count,const T* source=nullptr) {
  CK(cudaMalloc((void**)&p,count*sizeof(T)));allocations.push_back(p);
  if(source) CK(cudaMemcpy(p,source,count*sizeof(T),cudaMemcpyHostToDevice)); else CK(cudaMemset(p,0,count*sizeof(T)));
 }
 ~Net(){for(auto p:allocations)cudaFree(p);}
};
__inline__ __device__ float warp_sum(float v){for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);return v;}
__global__ void mv(int n,const int* ptr,const int* idx,const float* w,const float* h,float* out){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(row<n){float s=0;for(int e=ptr[row]+lane;e<ptr[row+1];e+=32)s+=w[e]*h[idx[e]];s=warp_sum(s);if(lane==0)out[row]=s;}
}
__global__ void mtv(int n,const int* ptr,const int* order,const int* post,const float* w,const float* g,float* out){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(row<n){float s=0;for(int k=ptr[row]+lane;k<ptr[row+1];k+=32){int e=order[k];s+=w[e]*g[post[e]];}s=warp_sum(s);if(lane==0)out[row]=s;}
}
__global__ void integrate(int n,const float* prev,const float* z,const float* obs,const int* ch,const float* sign,float* next,float* t,unsigned char* seen){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float a=tanhf(z[i]+(ch[i]<0?0:obs[ch[i]]*sign[i]));next[i]=.5f*prev[i]+.5f*a;if(t)t[i]=a;if(seen&&fabsf(next[i])>1e-6f)seen[i]=1;}
}
__global__ void readout(int n,const float* h,const float* r,const float* b,float* q,float* norm){
 __shared__ float sums[256];int a=blockIdx.x;float v=0;
 for(int i=threadIdx.x;i<n;i+=256)v+=a<4?r[a*n+i]*h[i]:h[i]*h[i];
 sums[threadIdx.x]=v;__syncthreads();for(int d=128;d;d/=2){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
 if(!threadIdx.x){if(a<4)q[a]=sums[0]+b[a];else *norm=sums[0];}
}
__global__ void gradient2(int n,const float* r,int a,const float* t,float* gz){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)gz[i]=.5f*r[a*n+i]*(1-t[i]*t[i]);}
__global__ void gradient1(int n,const float* r,int a,const float* back,const float* t,float* gz){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)gz[i]=.5f*(.5f*r[a*n+i]+back[i])*(1-t[i]*t[i]);}
__global__ void td(const float* q,int a,float target,float* delta){*delta=fminf(1.f,fmaxf(-1.f,target-q[a]));}
__global__ void updatew(int m,const int* pre,const int* post,const float* h0,const float* h1,const float* gz1,const float* gz2,const float* base,float* w,const float* delta){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<m){int i=post[e],j=pre[e];float v=w[e]+.01f*(*delta)*(gz1[i]*h0[j]+gz2[i]*h1[j]);w[e]=fminf(fmaxf(0.f,2*base[e]),fmaxf(fminf(0.f,2*base[e]),v));}
}
__global__ void updater(int n,const float* h,float* r,float* b,int a,const float* delta,const float* norm){int i=blockIdx.x*blockDim.x+threadIdx.x;float rate=.25f*(*delta)/(1+*norm);if(i<n)r[a*n+i]+=rate*h[i];if(i==0)b[a]+=rate;}
static void reset(Net* s){CK(cudaMemset(s->h2,0,s->n*sizeof(float)));CK(cudaMemset(s->seen,0,s->n));s->cache=false;}
extern "C" {
const char* fs_error(){return error.c_str();}
void* fs_create(int n,int m,const int* ptr,const int* idx,const int* post,const int* tp,const int* order,const float* base,const int* channel,const float* sign,const float* r,const float* b,const float*){
 Net* s=new Net();try {
 s->n=n;s->m=m;s->alloc(s->ptr,n+1,ptr);s->alloc(s->idx,m,idx);s->alloc(s->post,m,post);s->alloc(s->tp,n+1,tp);s->alloc(s->order,m,order);
 s->alloc(s->base,m,base);s->alloc(s->w,m,base);s->alloc(s->channel,n,channel);s->alloc(s->sign,n,sign);s->alloc(s->r,4*n,r);s->alloc(s->b,4,b);
 for(float** p:{&s->h0,&s->h1,&s->h2,&s->t1,&s->t2,&s->p0,&s->p1,&s->p2,&s->z,&s->gz1,&s->gz2,&s->back})s->alloc(*p,n);
 s->alloc(s->obs,24);s->alloc(s->q,4);s->alloc(s->qp,4);s->alloc(s->norm,1);s->alloc(s->delta,1);s->alloc(s->seen,n);reset(s);return s;
 }catch(std::exception& e){error=e.what();delete s;return nullptr;}
}
int fs_reset(void* h){try{reset((Net*)h);return 0;}catch(std::exception& e){error=e.what();return 1;}}
int fs_step(void* h,const float* obs,int preview,float* out){try {
 Net* s=(Net*)h;int n=s->n;CK(cudaMemcpy(s->obs,obs,24*sizeof(float),cudaMemcpyHostToDevice));
 float *a=preview?s->p0:s->h0,*b=preview?s->p1:s->h1,*c=preview?s->p2:s->h2;
 CK(cudaMemcpy(a,s->h2,n*sizeof(float),cudaMemcpyDeviceToDevice));
 mv<<<(n+7)/8,256>>>(n,s->ptr,s->idx,s->w,a,s->z);
 integrate<<<(n+255)/256,256>>>(n,a,s->z,s->obs,s->channel,s->sign,b,preview?nullptr:s->t1,preview?nullptr:s->seen);
 mv<<<(n+7)/8,256>>>(n,s->ptr,s->idx,s->w,b,s->z);
 integrate<<<(n+255)/256,256>>>(n,b,s->z,s->obs,s->channel,s->sign,c,preview?nullptr:s->t2,preview?nullptr:s->seen);
 // Preview must not overwrite the norm used by the current observation's update.
 readout<<<preview?4:5,256>>>(n,c,s->r,s->b,preview?s->qp:s->q,s->norm);
 CK(cudaGetLastError());CK(cudaMemcpy(out,preview?s->qp:s->q,4*sizeof(float),cudaMemcpyDeviceToHost));if(!preview)s->cache=true;return 0;
 }catch(std::exception& e){error=e.what();return 1;}}
int fs_learn(void* h,int action,float target,int plastic,float* out){try {
 Net* s=(Net*)h;int n=s->n;if(!s->cache)throw std::runtime_error("step required before learning");
 td<<<1,1>>>(s->q,action,target,s->delta);
 if(plastic){
 gradient2<<<(n+255)/256,256>>>(n,s->r,action,s->t2,s->gz2);
 mtv<<<(n+7)/8,256>>>(n,s->tp,s->order,s->post,s->w,s->gz2,s->back);
 gradient1<<<(n+255)/256,256>>>(n,s->r,action,s->back,s->t1,s->gz1);
 updatew<<<(s->m+255)/256,256>>>(s->m,s->idx,s->post,s->h0,s->h1,s->gz1,s->gz2,s->base,s->w,s->delta);
 }
 updater<<<(n+255)/256,256>>>(n,s->h2,s->r,s->b,action,s->delta,s->norm);
 CK(cudaGetLastError());CK(cudaMemcpy(out,s->delta,sizeof(float),cudaMemcpyDeviceToHost));s->cache=false;return 0;
 }catch(std::exception& e){error=e.what();return 1;}}
int fs_get(void* h,float* w,float* r,float* b){try{Net*s=(Net*)h;CK(cudaMemcpy(w,s->w,s->m*sizeof(float),cudaMemcpyDeviceToHost));CK(cudaMemcpy(r,s->r,4*s->n*sizeof(float),cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,s->b,4*sizeof(float),cudaMemcpyDeviceToHost));return 0;}catch(std::exception&e){error=e.what();return 1;}}
int fs_set(void* h,const float* w,const float* r,const float* b){try{Net*s=(Net*)h;CK(cudaMemcpy(s->w,w,s->m*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(s->r,r,4*s->n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(s->b,b,4*sizeof(float),cudaMemcpyHostToDevice));s->cache=false;return 0;}catch(std::exception&e){error=e.what();return 1;}}
int fs_state(void* h,float* a,unsigned char* seen){try{Net*s=(Net*)h;CK(cudaMemcpy(a,s->h2,s->n*sizeof(float),cudaMemcpyDeviceToHost));CK(cudaMemcpy(seen,s->seen,s->n,cudaMemcpyDeviceToHost));return 0;}catch(std::exception&e){error=e.what();return 1;}}
void fs_destroy(void* h){delete (Net*)h;}
}
