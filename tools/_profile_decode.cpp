#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "common/engine_config.h"
#include "model/qwen3_5_int4_model.h"
#include "model/tokenizer_hf.h"
#include "weights/qlwc_store.h"
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
int main(){
  auto cfg=llmoc::EngineConfig::load("configs/engine_int4.yaml");
  auto tok_dir=cfg.resolve_tokenizer_dir();
  llmoc::qlwc::QlwcStore store; store.open(cfg.model_path);
  llmoc::model::HfTokenizer tok; tok.load(tok_dir+"/tokenizer.json");
  llmoc::model::Qwen35Int4Model model; model.load(&store, tok_dir+"/config.json");
  auto ids=tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  llmoc::model::SessionCache cache; model.init_cache(cache, 4096);
  std::vector<float> logits;
  auto t0=Clock::now();
  model.forward(ids, cache, logits, true);
  auto t1=Clock::now();
  printf("prefill n=%zu %.1f ms\n", ids.size(), ms(t0,t1));
  double sum=0; int N=16;
  for(int i=0;i<N;++i){
    int32_t next=0; float best=-1e30f;
    for(int v=0;v<(int)logits.size();++v) if(logits[v]>best){best=logits[v];next=v;}
    auto a=Clock::now();
    model.forward({next}, cache, logits, false);
    auto b=Clock::now();
    double d=ms(a,b); sum+=d;
    printf("decode[%d] tok=%d %.2f ms\n", i, next, d);
  }
  printf("avg decode %.2f ms  => %.2f tok/s\n", sum/N, 1000.0/(sum/N));
  return 0;
}
