#include <immintrin.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
int main(){
 constexpr int width=512,hidden=4,batch=8192,lanes=32,groups=batch/lanes;
 std::vector<unsigned short>x(width*batch),w(width*hidden+hidden+1),out(batch);
 std::ifstream a("evidence/owner0.bin",std::ios::binary),b("evidence/owner1.bin",std::ios::binary);
 a.read((char*)x.data(),x.size()*2);b.read((char*)w.data(),w.size()*2);
 if(!a||!b)return 1;
 std::vector<int> indices[hidden];
 for(int h=0;h<hidden;h++)for(int j=0;j<width;j++)if(w[j*hidden+h])indices[h].push_back(j);
 for(bool sparse:{false,true}) {
 std::vector<double> timings;
 for(int r=0;r<110;r++){
 auto start=std::chrono::steady_clock::now();
 for(int g=0;g<groups;g++){
 __m512i score=_mm512_set1_epi16(w.back());
 for(int h=0;h<hidden;h++){
 __m512i sum=_mm512_set1_epi16(w[width*hidden+h]);
 if(sparse){for(int j:indices[h])sum=_mm512_add_epi16(sum,_mm512_mullo_epi16(_mm512_loadu_si512(x.data()+(g*width+j)*lanes),_mm512_set1_epi16(w[j*hidden+h])));}
 else{for(int j=0;j<width;j++)sum=_mm512_add_epi16(sum,_mm512_mullo_epi16(_mm512_loadu_si512(x.data()+(g*width+j)*lanes),_mm512_set1_epi16(w[j*hidden+h])));}
 auto square=_mm512_mullo_epi16(sum,sum);
 score=h<hidden/2?_mm512_add_epi16(score,square):_mm512_sub_epi16(score,square);
 }
 _mm512_storeu_si512(out.data()+g*lanes,_mm512_srli_epi16(score,15));
 }
 asm volatile(""::"r"(out.data()):"memory");
 if(r>=10)timings.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
 }
 std::sort(timings.begin(),timings.end());std::cout<<(sparse?"sparse":"dense")<<"_int16_avx512_ms "<<timings[50]<<"\n";
 std::ofstream o(sparse?"evidence/native_sparse_predictions.bin":"evidence/native_predictions.bin",std::ios::binary);o.write((char*)out.data(),out.size()*2);
 }
}
