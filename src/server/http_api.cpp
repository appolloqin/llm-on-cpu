// llm-on-cpu :: server/http_api.cpp
#include "server/http_api.h"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "common/log.h"
#include "model/tokenizer_hf.h"

namespace llmoc::server {
namespace {

std::vector<uint8_t> b64_decode(const std::string& in) {
  auto dec = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  std::vector<uint8_t> out;
  out.reserve(in.size() * 3 / 4);
  int val = 0, valb = -8;
  for (unsigned char c : in) {
    if (c == '=' || c <= ' ') continue;
    const int d = dec(c);
    if (d < 0) continue;
    val = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

bool parse_data_url_image(const std::string& url, std::vector<uint8_t>& out) {
  const auto pos = url.find("base64,");
  if (pos == std::string::npos) return false;
  out = b64_decode(url.substr(pos + 7));
  return !out.empty();
}

std::string json_safe_utf8(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = 0;
    if (c < 0x80)
      n = 1;
    else if ((c >> 5) == 0x6)
      n = 2;
    else if ((c >> 4) == 0xE)
      n = 3;
    else if ((c >> 3) == 0x1E)
      n = 4;
    else {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    if (i + n > s.size()) {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    bool ok = true;
    for (size_t k = 1; k < n; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) >> 6) != 0x2) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    out.append(s, i, n);
    i += n;
  }
  return out;
}

nlohmann::json token_logprob_to_json(const model::TokenLogprob& t) {
  nlohmann::json top = nlohmann::json::array();
  for (const auto& x : t.top_logprobs) {
    top.push_back({{"token", json_safe_utf8(x.token)},
                   {"logprob", x.logprob},
                   {"bytes", x.bytes}});
  }
  return {{"token", json_safe_utf8(t.token)},
          {"logprob", t.logprob},
          {"bytes", t.bytes},
          {"top_logprobs", std::move(top)}};
}

nlohmann::json logprobs_content_json(const std::vector<model::TokenLogprob>& lps) {
  nlohmann::json content = nlohmann::json::array();
  for (const auto& t : lps) content.push_back(token_logprob_to_json(t));
  return {{"content", std::move(content)}};
}

model::ChatMessage parse_chat_message(const nlohmann::json& m) {
  model::ChatMessage msg;
  msg.role = m.value("role", "user");
  const auto& content = m.contains("content") ? m["content"] : nlohmann::json("");
  if (content.is_string()) {
    msg.content = content.get<std::string>();
  } else if (content.is_array()) {
    for (const auto& part : content) {
      if (!part.is_object()) continue;
      const std::string type = part.value("type", "");
      if (type == "text") {
        msg.content += part.value("text", "");
      } else if (type == "image_url" && part.contains("image_url")) {
        std::string url;
        if (part["image_url"].is_string())
          url = part["image_url"].get<std::string>();
        else
          url = part["image_url"].value("url", "");
        std::vector<uint8_t> bytes;
        if (parse_data_url_image(url, bytes))
          msg.images.push_back({std::move(bytes)});
        else
          throw std::runtime_error(
              "image_url must be data:image/...;base64,... (blob: URLs are not readable by server)");
      }
    }
  } else if (!content.is_null()) {
    msg.content = content.dump();
  }
  return msg;
}

}  // namespace

void HttpApi::bind(const EngineConfig& cfg, sched::Scheduler* sched) {
  cfg_ = cfg;
  sched_ = sched;
  if (const char* k = std::getenv(cfg_.api_key_env.c_str())) api_key_ = k;
}

void HttpApi::listen() {
  httplib::Server svr;

  auto auth_ok = [&](const httplib::Request& req) {
    if (api_key_.empty()) return true;
    const auto auth = req.get_header_value("Authorization");
    return auth == ("Bearer " + api_key_);
  };

  svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  // 简易对话页（此前只有 API，打开 / 会 404）
  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    static constexpr const char* kHtml = R"HTML(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>llm-on-cpu</title>
<style>
:root{--bg:#0f1419;--panel:#1a2332;--text:#e7ecf3;--muted:#8b9bb4;--accent:#3d8bfd;--border:#2a3548}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
header{padding:14px 18px;border-bottom:1px solid var(--border);font-weight:600;letter-spacing:.02em}
#log{flex:1;overflow:auto;padding:16px 18px;display:flex;flex-direction:column;gap:12px}
/* 气泡随内容长高；只滚动 #log，不在每条消息上套滚动条 */
.msg{max-width:min(720px,92%);padding:10px 14px;border-radius:10px;word-break:break-word}
.msg .body{white-space:pre-wrap}
.user{align-self:flex-end;background:#243044}.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--border)}
.meta{color:var(--muted);font-size:12px;margin-bottom:4px}
.muted{color:var(--muted);font-style:italic}
.msg img.att{display:block;max-width:min(420px,100%);max-height:360px;width:auto;height:auto;border-radius:8px;margin:6px 0;object-fit:contain;background:#0a0e14}
.atts{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
#composer{border-top:1px solid var(--border);background:#121821}
#preview{display:none;flex-wrap:wrap;gap:8px;padding:8px 18px 0}
#preview.show{display:flex}
#preview .chip{position:relative}
#preview img{max-height:72px;max-width:120px;border-radius:6px;display:block;border:1px solid var(--border)}
#preview button.x{position:absolute;top:-6px;right:-6px;width:20px;height:20px;padding:0;border-radius:50%;font-size:12px;line-height:1;background:#c44;border:0;color:#fff;cursor:pointer}
form{display:flex;gap:8px;padding:12px 18px;align-items:flex-end}
input,button,textarea{font:inherit}
#q{flex:1;min-height:42px;max-height:min(40vh,320px);padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);resize:none;overflow-y:hidden;line-height:1.4;field:vertical}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
button.ghost{background:transparent;border:1px solid var(--border);color:var(--text)}
#status{padding:0 18px 10px;color:var(--muted);font-size:12px}
.think{color:var(--muted);font-size:12px;border-left:2px solid var(--border);padding-left:10px;margin-bottom:8px;white-space:pre-wrap}
#file{display:none}
</style>
</head>
<body>
<header>llm-on-cpu · 本地对话</header>
<div id="log"></div>
<div id="status">CPU 推理较慢，请耐心等待。</div>
<div id="composer">
  <div id="preview"></div>
  <form id="f">
    <button type="button" class="ghost" id="pick" title="添加图片">图片</button>
    <input id="file" type="file" accept="image/*" multiple/>
    <textarea id="q" placeholder="输入消息，可粘贴或添加图片…" rows="1"></textarea>
    <button id="go" type="submit">发送</button>
  </form>
</div>
<script>
const log=document.getElementById('log'), q=document.getElementById('q'), go=document.getElementById('go'), st=document.getElementById('status');
const preview=document.getElementById('preview'), file=document.getElementById('file');
const history=[];
let pendingImgs=[]; // {url: dataURL, name}
function autosizeQ(){
  // 内容变多时拉高输入框；只有顶到上限才出现滚动
  const max=Math.min(Math.floor(window.innerHeight*0.4),320);
  q.style.overflowY='hidden';
  q.style.height='auto';
  const need=q.scrollHeight;
  const h=Math.max(42, Math.min(need, max));
  q.style.height=h+'px';
  if(need>max) q.style.overflowY='auto';
}
q.addEventListener('input',autosizeQ);
window.addEventListener('resize',autosizeQ);
autosizeQ();
function stripThink(t){
  t=String(t||'');
  const close='</think>';
  const re=/<think>([\s\S]*?)<\/think>/g;
  let m, answer=t, paired=false;
  while((m=re.exec(t))){paired=true;}
  if(paired){
    answer=t.replace(/<think>[\s\S]*?<\/think>/g,'').replace(/^\s+/,'').trim();
  }else{
    const c=t.indexOf(close);
    if(c>=0){
      // 预填空 think 后模型仍可能输出「草稿…</think>正文」——丢弃闭合前草稿
      answer=t.slice(c+close.length).replace(/^\s+/,'').trim();
    }
  }
  // 聊天页 enable_thinking:false，不展示思考区，避免与正文重复
  return {thinking:'',answer};
}
function normalize(t){
  t=String(t||'').replace(/\r\n/g,'\n').replace(/[\u200b\ufeff]/g,'');
  return t.replace(/\n{3,}/g,'\n\n').trim();
}
function fileToDataUrl(f){
  return new Promise((resolve,reject)=>{
    const r=new FileReader();
    r.onload=()=>resolve(String(r.result||''));
    r.onerror=()=>reject(r.error||new Error('图片读取失败'));
    r.readAsDataURL(f);
  });
}
function renderPreview(){
  preview.innerHTML='';
  if(!pendingImgs.length){preview.classList.remove('show');return;}
  preview.classList.add('show');
  pendingImgs.forEach((im,i)=>{
    const chip=document.createElement('div');chip.className='chip';
    const img=document.createElement('img');img.src=im.url;img.alt=im.name||'image';
    const x=document.createElement('button');x.type='button';x.className='x';x.textContent='×';
    x.onclick=()=>{pendingImgs.splice(i,1);renderPreview();};
    chip.appendChild(img);chip.appendChild(x);preview.appendChild(chip);
  });
}
async function addFiles(list){
  for(const f of list){
    if(!f.type || !f.type.startsWith('image/'))continue;
    try{
      const url=await fileToDataUrl(f);
      if(!url.startsWith('data:image/'))throw new Error('非图片 data URL');
      pendingImgs.push({url,name:f.name||'image'});
    }catch(err){st.textContent='添加图片失败: '+(err.message||err);}
  }
  renderPreview();
}
document.getElementById('pick').onclick=()=>file.click();
file.onchange=async()=>{await addFiles(file.files||[]);file.value='';};
q.addEventListener('paste',async(e)=>{
  const items=e.clipboardData&&e.clipboardData.items;if(!items)return;
  const files=[];
  for(const it of items){if(it.type.startsWith('image/')){const f=it.getAsFile();if(f)files.push(f);}}
  if(files.length){e.preventDefault();await addFiles(files);}
});
q.addEventListener('dragover',(e)=>{e.preventDefault();});
q.addEventListener('drop',async(e)=>{
  e.preventDefault();
  await addFiles(e.dataTransfer&&e.dataTransfer.files||[]);
});
function setBody(el,text,imgs){
  const {thinking,answer}=stripThink(text);
  const n=normalize(answer);
  let th=el.querySelector('.think');
  let body=el.querySelector('.body');
  let atts=el.querySelector('.atts');
  if(!body){body=document.createElement('div');body.className='body';el.appendChild(body);}
  if(thinking){
    if(!th){th=document.createElement('div');th.className='think';el.insertBefore(th,body);}
    th.textContent='思考：'+thinking;
  }else if(th){th.remove();}
  if(imgs&&imgs.length){
    if(!atts){atts=document.createElement('div');atts.className='atts';el.insertBefore(atts,body);}
    atts.innerHTML='';
    imgs.forEach(im=>{const img=document.createElement('img');img.className='att';img.src=im.url||im;img.alt='';atts.appendChild(img);});
  }else if(atts){atts.remove();}
  if(!n && !(imgs&&imgs.length)){body.className='body muted';body.textContent='（无有效文本）';}
  else if(!n){body.className='body muted';body.textContent='';}
  else{body.className='body';body.textContent=n;}
}
function add(role,text,imgs){
  const d=document.createElement('div');
  d.className='msg '+(role==='user'?'user':'bot');
  d.innerHTML='<div class="meta">'+(role==='user'?'你':'助手')+'</div>';
  setBody(d,text,imgs);
  log.appendChild(d);log.scrollTop=log.scrollHeight;return d;
}
document.getElementById('f').onsubmit=async(e)=>{
  e.preventDefault();
  const text=q.value.trim();
  const imgs=pendingImgs.slice();
  if(!text && !imgs.length)return;
  q.value='';
  autosizeQ();
  pendingImgs=[];renderPreview();
  // 必须发 data:image/...;base64,...（blob: 服务端读不到）
  const content=[];
  for(const im of imgs){
    if(!im.url || !String(im.url).startsWith('data:image/')){
      setBody(add('assistant',''),'错误: 图片未转成 data URL，请重新添加');
      return;
    }
    content.push({type:'image_url',image_url:{url:im.url}});
  }
  if(text) content.push({type:'text',text:text});
  const userMsg={role:'user',content: content.length? content : text};
  history.push(userMsg);
  add('user',text,imgs);
  const bot=add('assistant','…');go.disabled=true;st.textContent=imgs.length?'视觉编码中…':'生成中…';
  try{
    const payload={messages:history,max_tokens:2048,stream:true,temperature:0,enable_thinking:false};
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)});
    if(!r.ok)throw new Error(await r.text());
    const reader=r.body.getReader(),dec=new TextDecoder();let acc='',buf='';
    while(true){
      const {done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\n\n');buf=parts.pop();
      for(const p of parts){
        const line=p.trim();if(!line.startsWith('data:'))continue;
        const raw=line.slice(5).trim();if(raw==='[DONE]')continue;
        try{
          const j=JSON.parse(raw);
          if(j.error){acc='错误: '+(typeof j.error==='string'?j.error:JSON.stringify(j.error));setBody(bot,acc);continue;}
          const d=j.choices?.[0]?.delta?.content;if(d){acc+=d;setBody(bot,acc);log.scrollTop=log.scrollHeight;}
        }catch(_){}
      }
    }
    if(!stripThink(acc).answer.trim()){
      const r2=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({...payload,stream:false})});
      const j=await r2.json();
      if(j.error) acc='错误: '+(typeof j.error==='string'?j.error:JSON.stringify(j.error));
      else acc=j.choices?.[0]?.message?.content||'';
      setBody(bot,acc);
    }
    const clean=normalize(stripThink(acc).answer)||'(empty)';
    history.push({role:'assistant',content:clean});
    st.textContent='就绪';
  }catch(err){setBody(bot,'错误: '+err.message);st.textContent='失败';}
  finally{go.disabled=false;q.focus();}
};
fetch('/healthz').then(r=>r.json()).then(()=>st.textContent='服务正常 · 已接入视觉（CPU，图会先缩到约 256²）').catch(()=>st.textContent='无法连接 /healthz');
</script>
</body>
</html>)HTML";
    res.set_content(kHtml, "text/html; charset=utf-8");
  });

  svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    const auto m = sched_->metrics();
    std::ostringstream oss;
    oss << "# TYPE llmoc_requests_total counter\n"
        << "llmoc_requests_total " << m.requests_total << "\n"
        << "# TYPE llmoc_tokens_generated_total counter\n"
        << "llmoc_tokens_generated_total " << m.tokens_generated << "\n"
        << "# TYPE llmoc_prompt_tokens_total counter\n"
        << "llmoc_prompt_tokens_total " << m.prompt_tokens << "\n"
        << "# TYPE llmoc_last_tps gauge\n"
        << "llmoc_last_tps " << m.last_tps << "\n"
        << "# TYPE llmoc_queue_depth gauge\n"
        << "llmoc_queue_depth " << m.queue_depth << "\n"
        << "# TYPE llmoc_radix_nodes gauge\n"
        << "llmoc_radix_nodes " << m.radix_nodes << "\n";
    res.set_content(oss.str(), "text/plain; version=0.0.4");
  });

  svr.Post("/v1/chat/completions", [this, &auth_ok](const httplib::Request& req,
                                                    httplib::Response& res) {
    if (!auth_ok(req)) {
      res.status = 401;
      res.set_content(R"({"error":"unauthorized"})", "application/json");
      return;
    }
    nlohmann::json body;
    try {
      body = nlohmann::json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"error":"invalid json"})", "application/json");
      return;
    }
    model::GenerateRequest greq;
    greq.max_new_tokens = body.value("max_tokens", cfg_.max_new_tokens);
    greq.stream = body.value("stream", false);
    greq.temperature = body.value("temperature", 0.0f);
    greq.enable_thinking = body.value("enable_thinking", false);
    greq.mtp = body.value("mtp", cfg_.mtp);
    greq.spec_k = body.value("spec_k", cfg_.spec_k);
    greq.logprobs = body.value("logprobs", false);
    greq.top_logprobs = body.value("top_logprobs", 0);
    if (greq.logprobs && greq.top_logprobs < 0) greq.top_logprobs = 0;
    if (greq.top_logprobs > 20) greq.top_logprobs = 20;
    if (!body.contains("messages") || !body["messages"].is_array()) {
      res.status = 400;
      res.set_content(R"({"error":"messages required"})", "application/json");
      return;
    }
    for (const auto& m : body["messages"]) {
      greq.messages.push_back(parse_chat_message(m));
    }

    if (greq.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Content-Type", "text/event-stream");
      res.set_chunked_content_provider(
          "text/event-stream",
          [this, greq](size_t /*offset*/, httplib::DataSink& sink) mutable {
            try {
              const std::string id = "chatcmpl-llmoc";
              auto on_tok = [&](const std::string& delta, const model::TokenLogprob* lp) {
                nlohmann::json choice = {{"index", 0},
                                         {"delta", {{"content", delta}}},
                                         {"finish_reason", nullptr}};
                if (lp) {
                  choice["logprobs"] = {{"content", nlohmann::json::array({token_logprob_to_json(*lp)})}};
                }
                nlohmann::json chunk = {
                    {"id", id},
                    {"object", "chat.completion.chunk"},
                    {"choices", nlohmann::json::array({std::move(choice)})}};
                const std::string line =
                    "data: " +
                    chunk.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n\n";
                sink.write(line.data(), line.size());
              };
              auto result = sched_->enqueue_sync(greq, on_tok);
              nlohmann::json done_choice = {{"index", 0},
                                            {"delta", nlohmann::json::object()},
                                            {"finish_reason", "stop"}};
              nlohmann::json done = {{"id", id},
                                     {"object", "chat.completion.chunk"},
                                     {"choices", nlohmann::json::array({std::move(done_choice)})}};
              if (greq.logprobs && result.perplexity > 0.0) {
                done["usage"] = {{"prompt_tokens", result.prompt_tokens},
                                 {"completion_tokens", result.completion_tokens},
                                 {"total_tokens", result.prompt_tokens + result.completion_tokens},
                                 {"perplexity", result.perplexity}};
              }
              const std::string line = "data: " + done.dump() + "\n\n";
              sink.write(line.data(), line.size());
              const char* end = "data: [DONE]\n\n";
              sink.write(end, std::strlen(end));
              sink.done();
            } catch (const std::exception& e) {
              LOG_ERROR("chat stream failed: %s", e.what());
              nlohmann::json err = {{"error", e.what()}};
              const std::string line = "data: " + err.dump() + "\n\n";
              sink.write(line.data(), line.size());
              sink.done();
            }
            return true;
          });
      return;
    }

    try {
      auto result = sched_->enqueue_sync(greq);
      nlohmann::json choice = {{"index", 0},
                               {"message", {{"role", "assistant"}, {"content", result.text}}},
                               {"finish_reason", "stop"},
                               {"logprobs", nullptr}};
      if (greq.logprobs) choice["logprobs"] = logprobs_content_json(result.logprobs);
      nlohmann::json usage = {{"prompt_tokens", result.prompt_tokens},
                              {"completion_tokens", result.completion_tokens},
                              {"total_tokens", result.prompt_tokens + result.completion_tokens}};
      if (greq.logprobs) usage["perplexity"] = result.perplexity;
      nlohmann::json resp = {{"id", "chatcmpl-llmoc"},
                             {"object", "chat.completion"},
                             {"model", "default"},
                             {"choices", nlohmann::json::array({std::move(choice)})},
                             {"usage", std::move(usage)}};
      res.set_content(resp.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
                      "application/json");
    } catch (const std::exception& e) {
      LOG_ERROR("chat failed: %s", e.what());
      res.status = 500;
      nlohmann::json err = {{"error", e.what()}};
      res.set_content(err.dump(), "application/json");
    }
  });

  LOG_INFO("listening on 0.0.0.0:%d", cfg_.server_port);
  if (!svr.listen("0.0.0.0", cfg_.server_port))
    throw std::runtime_error("failed to listen on port " + std::to_string(cfg_.server_port));
}

}  // namespace llmoc::server
