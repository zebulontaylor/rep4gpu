#include <immintrin.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
int main(){
 constexpr int width=513,batch=8192,lanes=32,groups=batch/lanes;
 std::vector<unsigned short>x(width*batch),w(width),out(batch);
 std::ifstream a("evidence/owner0.bin",std::ios::binary),b("evidence/owner1.bin",std::ios::binary);
 a.read((char*)x.data(),x.size()*2);b.read((char*)w.data(),w.size()*2);
 std::vector<double> timings;
 for(int r=0;r<110;r++){
 auto start=std::chrono::steady_clock::now();
 for(int g=0;g<groups;g++){
 __m512i sum=_mm512_setzero_si512();
 for(int j=0;j<width;j++)sum=_mm512_add_epi16(sum,_mm512_mullo_epi16(_mm512_loadu_si512(x.data()+(g*width+j)*lanes),_mm512_set1_epi16(w[j])));
 _mm512_storeu_si512(out.data()+g*lanes,_mm512_srli_epi16(sum,15));
 }
 asm volatile(""::"r"(out.data()):"memory");
 if(r>=10)timings.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
 }
 std::sort(timings.begin(),timings.end());std::cout<<"native_int16_avx512_ms "<<timings[50]<<"\n";
 std::ofstream o("evidence/native_predictions.bin",std::ios::binary);o.write((char*)out.data(),out.size()*2);
}
