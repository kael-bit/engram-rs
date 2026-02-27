const A=location.origin;
let allMem=[],stats={},health={},curView='all',page=0;
const PP=50;
let sortBy='newest';

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function ln(l){return['','Buffer','Working','Core'][l]||'?';}
function lc(n){return{buffer:1,working:2,core:3}[n]||0;}
function ago(ms){
  const s=(Date.now()-ms)/1e3;
  if(s<60)return Math.floor(s)+'s';
  if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h';
  return Math.floor(s/86400)+'d';
}
function toast(msg,ok=true){
  const el=document.createElement('div');
  el.className='toast '+(ok?'ok':'err');
  el.textContent=msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(()=>el.remove(),3000);
}

async function api(path,opts={}){
  const tk=localStorage.getItem('engram_token')||'';
  const ns=localStorage.getItem('engram_namespace')||'';
  const h={'Content-Type':'application/json',...opts.headers};
  if(tk)h['Authorization']='Bearer '+tk;
  if(ns&&ns!=='default')h['X-Namespace']=ns;
  const res=await fetch(A+path,{...opts,headers:h});
  if(res.status===401){
    const t=prompt('Auth token:');
    if(t){localStorage.setItem('engram_token',t);return api(path,opts);}
    throw new Error('Unauthorized');
  }
  if(res.status===204)return{};
  const text=await res.text();
  try{return JSON.parse(text);}catch{return text;}
}

/* --- Namespace management --- */
let knownNs=['default'];

function getCurNs(){return localStorage.getItem('engram_namespace')||'default';}

function updateNsUI(){
  const ns=getCurNs();
  document.getElementById('nsCurrent').textContent=ns;
  const sel=document.getElementById('nsSelect');
  sel.value=knownNs.includes(ns)?ns:'__custom__';
}

function updateNsList(statsData){
  if(statsData&&statsData.namespaces){
    const names=Array.isArray(statsData.namespaces)?statsData.namespaces:Object.keys(statsData.namespaces);
    names.sort();
    if(names.length) knownNs=names;
  }
  const sel=document.getElementById('nsSelect');
  const cur=getCurNs();
  sel.innerHTML=knownNs.map(n=>'<option value="'+esc(n)+'">'+esc(n)+'</option>').join('')+
    '<option value="__custom__">✚ Custom...</option>';
  if(knownNs.includes(cur)) sel.value=cur;
  else sel.value='__custom__';
  updateNsUI();
}

function onNsSelect(v){
  const inp=document.getElementById('nsInput');
  if(v==='__custom__'){
    inp.classList.add('show');inp.value='';inp.focus();
    return;
  }
  inp.classList.remove('show');
  setNamespace(v);
}

function applyCustomNs(){
  const v=document.getElementById('nsInput').value.trim();
  if(!v)return;
  document.getElementById('nsInput').classList.remove('show');
  if(!knownNs.includes(v))knownNs.push(v);
  const sel=document.getElementById('nsSelect');
  sel.innerHTML=knownNs.map(n=>'<option value="'+esc(n)+'">'+esc(n)+'</option>').join('')+
    '<option value="__custom__">✚ Custom...</option>';
  setNamespace(v);
}

function setNamespace(ns){
  const prev=getCurNs();
  if(ns===prev)return;
  localStorage.setItem('engram_namespace',ns);
  updateNsUI();
  toast('Namespace → '+ns);
  refresh();
}

// Restore saved namespace on load
(function initNs(){
  const saved=getCurNs();
  if(saved&&saved!=='default'){
    document.getElementById('nsCurrent').textContent=saved;
  }
})();

async function loadStats(){
  const[s,h]=await Promise.all([api('/stats'),api('/health')]);
  stats=s;health=h;
  document.getElementById('c-all').textContent=s.total;
  document.getElementById('c-buf').textContent=s.buffer;
  document.getElementById('c-wrk').textContent=s.working;
  document.getElementById('c-cor').textContent=s.core;
  document.getElementById('ver').textContent=' v'+(h.version||'?');
  // Fetch global stats (without namespace header) to get full namespace list
  try{
    const tk=localStorage.getItem('engram_token')||'';
    const gh={'Content-Type':'application/json'};
    if(tk)gh['Authorization']='Bearer '+tk;
    const gRes=await fetch(A+'/stats',{headers:gh});
    const gStats=await gRes.json();
    updateNsList(gStats);
  }catch(_){updateNsList(s);}
}

async function loadMem(){
  const d=await api('/memories?limit=10000');
  allMem=Array.isArray(d)?d:(d.memories||[]);
  allMem.sort((a,b)=>b.created_at-a.created_at);
  document.getElementById('c-prx').textContent=allMem.filter(m=>m.source==='proxy').length;
}

function card(m,score,rel){
  const tags=(m.tags||[]).map(t=>`<span class="tag">${esc(t)}</span>`).join(' ');
  const sc=score!=null?`<span class="score">${score.toFixed(3)}</span>`:'';
  const rl=rel!=null?`<span class="score" style="opacity:.6">rel ${rel.toFixed(3)}</span>`:'';
  const kind=m.kind?`<span class="kind-badge kind-${m.kind}">${m.kind}</span>`:`<span class="kind-badge kind-semantic">semantic</span>`;
  return`<div class="mc" data-id="${m.id}">
    <div class="ct">${esc((m.content||'').slice(0,280))}${m.content&&m.content.length>280?'…':''}</div>
    <div class="meta"><span class="lb l${m.layer}">${ln(m.layer)}</span>${sc}${rl}${kind}
      <span>imp ${(m.importance||0).toFixed(2)}</span><span>ac ${m.access_count||0}</span>${m.repetition_count?`<span>rep ${m.repetition_count}</span>`:''}
      <span>${ago(m.created_at)}</span>${tags}</div></div>`;
}

function recallCard(m){
  const tags=(m.tags||[]).map(t=>`<span class="tg">${esc(t)}</span>`).join('');
  const kind=m.kind?`<span class="kind-badge kind-${m.kind}">${m.kind}</span>`:`<span class="kind-badge kind-semantic">semantic</span>`;
  return`<div class="mc" style="cursor:default">
    <div class="ct">${esc((m.content||'').slice(0,280))}${m.content&&m.content.length>280?'…':''}</div>
    <div class="meta"><span class="lb l${m.layer}">${m.layer}</span><span class="score">${(m.score||0).toFixed(3)}</span>${kind}
      <span style="font-size:10px;color:#777">${m.id}</span>${tags}</div></div>`;
}

function getMem(){
  if(curView==='proxy')return allMem.filter(m=>m.source==='proxy');
  if(['buffer','working','core'].includes(curView))return allMem.filter(m=>m.layer===lc(curView));
  return allMem;
}
function memWeight(m){
  const rep=Math.min((m.repetition_count||0)*0.1,0.5);
  const ac=Math.min(Math.log(1+(m.access_count||0))*0.1,0.3);
  const kb=m.kind==='procedural'?1.3:m.kind==='episodic'?0.8:1.0;
  const lb=m.layer===3?1.2:m.layer===1?0.8:1.0;
  return((m.importance||0)+rep+ac)*kb*lb;
}
function sortMem(a){
  return[...a].sort((x,y)=>{
    if(sortBy==='newest')return y.created_at-x.created_at;
    if(sortBy==='oldest')return x.created_at-y.created_at;
    if(sortBy==='weight_high')return memWeight(y)-memWeight(x);
    if(sortBy==='weight_low')return memWeight(x)-memWeight(y);
    return y.created_at-x.created_at;
  });
}

function renderList(mems,scores){
  const M=document.getElementById('main');
  const sorted=scores?mems:sortMem(mems);
  const total=sorted.length;
  const paged=sorted.slice(page*PP,(page+1)*PP);
  const pages=Math.ceil(total/PP);
  M.innerHTML=`
    <h2>${{all:'All',buffer:'Buffer',working:'Working',core:'Core',proxy:'Proxy'}[curView]||'Memories'}</h2>
    <div class="stats">
      <div class="sc"><div class="v">${stats.total||0}</div><div class="l">Total</div></div>
      <div class="sc buf"><div class="v">${stats.buffer||0}</div><div class="l">Buffer</div></div>
      <div class="sc wrk"><div class="v">${stats.working||0}</div><div class="l">Working</div></div>
      <div class="sc cor"><div class="v">${stats.core||0}</div><div class="l">Core</div></div>
    </div>
    <div class="row">
      <input type="text" id="flt" placeholder="Filter..." style="flex:1" oninput="applyFilter()">
      <select id="sortSel" onchange="changeSort(this.value)" style="width:130px">
        <option value="newest" ${sortBy==='newest'?'selected':''}>Newest</option>
        <option value="oldest" ${sortBy==='oldest'?'selected':''}>Oldest</option>
        <option value="weight_high" ${sortBy==='weight_high'?'selected':''}>Weight ↓</option>
        <option value="weight_low" ${sortBy==='weight_low'?'selected':''}>Weight ↑</option>
      </select>
    </div>
    <div class="ml" id="memList">${paged.length?paged.map((m,i)=>card(m,scores?scores[page*PP+i]:null)).join(''):'<div class="empty">No memories</div>'}</div>
    ${pages>1?`<div class="pager">
      <button onclick="goPage(${page-1})" ${page===0?'disabled':''}>Prev</button>
      <span>${page+1}/${pages} (${total})</span>
      <button onclick="goPage(${page+1})" ${page>=pages-1?'disabled':''}>Next</button>
    </div>`:''}`;
  bindCards();
}

function goPage(p){page=Math.max(0,p);renderList(getMem());}
function changeSort(v){sortBy=v;page=0;renderList(getMem());}
function applyFilter(){
  const q=(document.getElementById('flt')||{}).value||'';
  let mems=getMem();
  if(q){const ql=q.toLowerCase();mems=mems.filter(m=>(m.content||'').toLowerCase().includes(ql)||(m.tags||[]).some(t=>t.toLowerCase().includes(ql))||m.id.startsWith(ql)||(m.kind||'').toLowerCase().includes(ql));}
  const el=document.getElementById('memList');
  if(el){el.innerHTML=sortMem(mems).slice(0,PP).map(m=>card(m)).join('')||'<div class="empty">No matches</div>';bindCards();}
}
function bindCards(){document.getElementById('main').querySelectorAll('.mc').forEach(c=>{c.onclick=()=>showDetail(c.dataset.id);});}

function showDetail(id){
  const m=allMem.find(x=>x.id===id);if(!m)return;
  const p=document.getElementById('detail');
  const imp=(m.importance*100).toFixed(0);
  p.innerHTML=`
    <button class="close-btn" onclick="closeDetail()">×</button>
    <h3 style="margin-bottom:14px"><span class="lb l${m.layer}">${ln(m.layer)}</span> Memory</h3>
    <div class="df"><div class="dl">ID</div><div class="dv" style="font-size:10px;color:#555">${m.id}</div></div>
    <div class="df"><div class="dl">Content</div>
      <textarea id="editContent" rows="6" style="width:100%">${esc(m.content)}</textarea>
    </div>
    <div class="df"><div class="dl">Importance (${imp}%)</div>
      <input type="range" id="editImp" min="0" max="1" step="0.05" value="${m.importance}" style="width:100%">
      <div class="bar"><div class="bar-fill" style="width:${imp}%;background:${m.importance>.7?'#4caf50':m.importance>.4?'#d4a44a':'#555'}"></div></div>
    </div>
    <div class="df"><div class="dl">Layer</div>
      <select id="editLayer" style="width:100%">
        <option value="1" ${m.layer===1?'selected':''}>Buffer</option>
        <option value="2" ${m.layer===2?'selected':''}>Working</option>
        <option value="3" ${m.layer===3?'selected':''}>Core</option>
      </select>
    </div>
    <div class="df"><div class="dl">Tags</div>
      <input type="text" id="editTags" value="${(m.tags||[]).join(', ')}" style="width:100%">
    </div>
    <div class="df"><div class="dl">Kind</div>
      <select id="editKind" style="width:100%">
        <option value="semantic" ${(m.kind||'semantic')==='semantic'?'selected':''}>semantic</option>
        <option value="episodic" ${m.kind==='episodic'?'selected':''}>episodic</option>
        <option value="procedural" ${m.kind==='procedural'?'selected':''}>procedural</option>
      </select>
    </div>
    <div class="df"><div class="dl">Source</div><div class="dv">${esc(m.source||'—')}</div></div>
    <div class="df"><div class="dl">Access / Repetition</div><div class="dv">${m.access_count||0} / ${m.repetition_count||0}</div></div>
    <div class="df"><div class="dl">Namespace</div><div class="dv">${esc(m.namespace||'default')}</div></div>
    <div class="df"><div class="dl">Created</div><div class="dv">${new Date(m.created_at).toLocaleString()}</div></div>
    <div class="df"><div class="dl">Last accessed</div><div class="dv">${new Date(m.last_accessed).toLocaleString()}</div></div>
    <div class="df"><div class="dl">Decay rate</div><div class="dv">${m.decay_rate}</div></div>
    <div class="detail-btns">
      <button class="btn" onclick="saveMem('${m.id}')">Save</button>
      <button class="btn btn-danger" onclick="deleteMem('${m.id}')">Delete</button>
    </div>`;
  p.classList.add('open');
  document.getElementById('panelBg').classList.add('open');
}

function closeDetail(){
  document.getElementById('detail').classList.remove('open');
  document.getElementById('panelBg').classList.remove('open');
}

async function saveMem(id){
  const ct=document.getElementById('editContent').value.trim();
  const imp=parseFloat(document.getElementById('editImp').value);
  const layer=parseInt(document.getElementById('editLayer').value);
  const tags=document.getElementById('editTags').value.split(',').map(t=>t.trim()).filter(Boolean);
  const kind=document.getElementById('editKind').value;
  try{
    await api('/memories/'+id,{method:'PATCH',body:JSON.stringify({content:ct,importance:imp,layer,tags,kind})});
    toast('Saved');closeDetail();await refresh();
  }catch(e){toast('Error: '+e.message,false);}
}

async function deleteMem(id){
  if(!confirm('Delete?'))return;
  await api('/memories/'+id,{method:'DELETE'});
  toast('Deleted');closeDetail();await refresh();
}

function renderRecall(){
  document.getElementById('main').innerHTML=`
    <h2>Recall</h2>
    <div class="card">
      <div class="row"><input type="text" id="rQ" placeholder="Query..." style="flex:1" onkeydown="if(event.key==='Enter')doRecall()">
        <button class="btn" onclick="doRecall()">Recall</button></div>
      <div class="row" style="margin:0">
        <label><input type="checkbox" id="rExp"> expand</label>
        <label>limit <input type="number" id="rLim" value="10" min="1" max="50" style="width:50px"></label>
        <label>min score <input type="number" id="rMin" value="0.3" min="0" max="1" step=".05" style="width:55px"></label>
        <input type="text" id="rTags" placeholder="tags" style="width:100px">
        <select id="rKind" style="width:90px"><option value="">any</option><option>semantic</option><option>episodic</option><option>procedural</option></select>
      </div>
    </div>
    <div id="rRes"><div class="empty">Enter a query</div></div>`;
}

async function doRecall(){
  const q=document.getElementById('rQ').value.trim();if(!q)return;
  const body={query:q,limit:parseInt(document.getElementById('rLim').value)||10,
    expand:document.getElementById('rExp').checked,min_score:parseFloat(document.getElementById('rMin').value)||0};
  const tags=document.getElementById('rTags').value.trim();
  const kind=document.getElementById('rKind').value;
  if(tags)body.tags=tags.split(',').map(t=>t.trim());
  if(kind)body.kind=kind;
  document.getElementById('rRes').innerHTML='<div class="empty">Searching...</div>';
  const t0=performance.now();
  const d=await api('/recall',{method:'POST',body:JSON.stringify(body)});
  const ms=(performance.now()-t0).toFixed(0);
  const r=d.memories||[];
  document.getElementById('rRes').innerHTML=`
    <div style="font-size:11px;color:#555;margin:10px 0 6px">${r.length} results, ${ms}ms — ${d.search_mode||'?'}</div>
    <div class="ml">${r.length?r.map(m=>recallCard(m)).join(''):'<div class="empty">Nothing found</div>'}</div>`;
}

function renderSearch(){
  document.getElementById('main').innerHTML=`
    <h2>Search</h2>
    <div class="card">
      <div class="row" style="margin:0"><input type="text" id="sQ" placeholder="Search text..." style="flex:1" onkeydown="if(event.key==='Enter')doSearch()">
        <button class="btn" onclick="doSearch()">Search</button></div>
    </div>
    <div id="sRes"><div class="empty">Full-text search</div></div>`;
}

function renderResume(){
  document.getElementById('main').innerHTML=`
    <h2>Resume</h2>
    <div class="card">
      <div class="row" style="margin:0;gap:12px;flex-wrap:wrap">
        <label>Epochs <input type="number" id="resEpochs" value="24" min="1" max="500" style="width:55px"></label>
        <label><input type="checkbox" id="resCompact"> compact</label>
        <label>Budget <input type="number" id="resBudget" value="0" min="0" style="width:70px" placeholder="0=unlimited"></label>
        <label>Format <select id="resFormat" style="padding:4px 8px;background:#222;color:#eee;border:1px solid #444;border-radius:4px"><option value="json">JSON</option><option value="text">Raw Text</option></select></label>
        <button class="btn" onclick="doResume()">Resume</button>
      </div>
    </div>
    <div id="resResult"><div class="empty">Hit Resume to see what an agent gets on wake-up</div></div>`;
}

let lastResumeData=null;

function parseTopicsString(s){
  if(!s||typeof s!=='string')return[];
  const results=[];
  const lines=s.trim().split('\n');
  for(const line of lines){
    const m=line.match(/^(\w+):\s*"(.+?)"\s*\[(\d+)\]$/);
    if(m)results.push({id:m[1],name:m[2],count:parseInt(m[3])});
  }
  return results;
}

function renderResumeResult(r,ms){
  const countLine='core: '+(r.core_count||0)+', working: '+(r.working_count||0)+
    ', buffer: '+(r.buffer_count||0)+', recent: '+(r.recent_count||0);

  function memCard(m){
    const tags=(m.tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join(' ');
    const layer=m.layer?('<span class="lb l'+m.layer+'">'+ln(m.layer)+'</span>'):'';
    const ts=m.created_at?new Date(m.created_at).toLocaleString():'';
    return '<div class="mc" style="cursor:default"><div class="ct">'+esc(m.content||'')+'</div>'+
      '<div class="meta">'+layer+' <span>'+ts+'</span> '+tags+'</div></div>';
  }

  function section(title,items){
    if(!items||!items.length) return '';
    return '<div class="resume-section"><b class="resume-section-hdr">'+title+' ('+items.length+')</b>'+
      '<div class="ml" style="margin-top:6px">'+items.map(m=>memCard(m)).join('')+'</div></div>';
  }

  let html='<div style="font-size:11px;color:#555;margin:10px 0 12px">'+ms+'ms — '+countLine+'</div>';
  html+=section('Core',r.core);
  html+=section('Recent',r.recent);

  // Topics section
  const topics=parseTopicsString(r.topics);
  if(topics.length){
    html+='<div class="resume-section"><b class="resume-section-hdr">Topics ('+topics.length+')</b>';
    html+='<div class="topic-grid" style="margin-top:6px">';
    for(const t of topics){
      html+='<div class="topic-card" onclick="expandResumeTopic(this,\''+esc(t.id)+'\')">';
      html+='<div class="tc-head"><span><span class="tc-id">'+esc(t.id)+'</span><span class="tc-name">'+esc(t.name)+'</span></span>';
      html+='<span class="tc-count">'+t.count+' memories</span></div>';
      html+='<div class="tc-body" style="display:none"></div>';
      html+='</div>';
    }
    html+='</div></div>';
  }

  // Triggers section
  const triggers=r.triggers||[];
  if(triggers.length){
    html+='<div class="resume-section"><b class="resume-section-hdr">Triggers ('+triggers.length+')</b>';
    html+='<div style="margin-top:6px">'+triggers.map(t=>'<span class="trigger-tag">'+esc(t)+'</span>').join('')+'</div></div>';
  }

  if(!html.includes('resume-section')&&!html.includes('ml')) html='<div class="empty">Empty resume</div>';
  document.getElementById('resResult').innerHTML=html;
}

async function doResume(){
  const epochs=parseInt(document.getElementById('resEpochs').value)||24;
  const compact=document.getElementById('resCompact').checked;
  const budget=parseInt(document.getElementById('resBudget').value)||0;
  const fmt=document.getElementById('resFormat').value;
  document.getElementById('resResult').innerHTML='<div class="empty">Loading...</div>';
  const t0=performance.now();

  if(fmt==='text'){
    let url='/resume?recent_epochs='+epochs;
    if(compact) url+='&compact=true';
    if(budget>0) url+='&budget='+budget;
    try{
      const resp=await api(url);
      const ms=Math.round(performance.now()-t0);
      const charCount=resp.length;
      const tokenEst=Math.round(charCount/3.5);
      document.getElementById('resResult').innerHTML=
        '<div style="font-size:11px;color:#555;margin:10px 0 8px">'+ms+'ms'+
        ' — ~'+tokenEst+' tokens ('+charCount+' chars)</div>'+
        '<pre style="background:#111;border:1px solid #222;border-radius:6px;padding:14px;'+
        'font-size:12px;line-height:1.5;color:#ccc;white-space:pre-wrap;word-break:break-word;'+
        'max-height:80vh;overflow-y:auto;font-family:monospace">'+esc(resp)+'</pre>';
    }catch(e){
      document.getElementById('resResult').innerHTML='<div class="empty">Error: '+e+'</div>';
    }
    return;
  }

  let url='/resume?format=json&recent_epochs='+epochs;
  if(compact) url+='&compact=true';
  if(budget>0) url+='&budget='+budget;
  const r=await api(url);
  const ms=Math.round(performance.now()-t0);
  lastResumeData={data:r,ms};
  renderResumeResult(r,ms);
}

async function expandResumeTopic(el,topicId){
  const body=el.querySelector('.tc-body');
  if(body.style.display!=='none'){body.style.display='none';el.classList.remove('expanded');return;}
  el.classList.add('expanded');
  body.style.display='block';
  body.innerHTML='<div class="empty">Loading...</div>';
  try{
    const tk=localStorage.getItem('engram_token')||'';
    const ns=localStorage.getItem('engram_namespace')||'';
    const h={'Content-Type':'application/json'};
    if(tk)h['Authorization']='Bearer '+tk;
    if(ns&&ns!=='default')h['X-Namespace']=ns;
    const res=await fetch(A+'/topic?touch=false',{method:'POST',headers:h,body:JSON.stringify({ids:[topicId]})});
    const d=await res.json();
    const mems=(d[topicId]&&d[topicId].memories)||[];
    if(!mems.length){body.innerHTML='<div class="empty">No memories</div>';return;}
    const tags=m=>(m.tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join(' ');
    body.innerHTML='<div class="ml">'+mems.map(m=>
      '<div class="mc" style="cursor:default"><div class="ct">'+esc((m.content||'').slice(0,300))+(m.content&&m.content.length>300?'…':'')+'</div>'+
      '<div class="meta"><span class="lb l'+(m.layer||1)+'">'+ln(m.layer||1)+'</span> <span>imp '+(m.importance||0).toFixed(2)+'</span> '+tags(m)+'</div></div>'
    ).join('')+'</div>';
  }catch(e){body.innerHTML='<div class="empty">Error: '+esc(e.message)+'</div>';}
}

/* --- Topics page --- */
let topicsData=[];

function renderTopics(){
  document.getElementById('main').innerHTML=`
    <h2>Topics</h2>
    <div id="topicsContent"><div class="empty">Loading topics...</div></div>`;
  loadTopics();
}

async function loadTopics(){
  try{
    const url='/resume?format=json&recent_epochs=500';
    const r=await api(url);
    topicsData=parseTopicsString(r.topics);
    if(!topicsData.length){
      document.getElementById('topicsContent').innerHTML='<div class="empty">No topics found</div>';
      return;
    }
    renderTopicCards();
  }catch(e){
    document.getElementById('topicsContent').innerHTML='<div class="empty">Error: '+esc(e.message)+'</div>';
  }
}

function renderTopicCards(){
  const el=document.getElementById('topicsContent');
  let html='<div style="font-size:11px;color:#555;margin-bottom:10px">'+topicsData.length+' topics</div>';
  html+='<div class="topic-grid">';
  for(const t of topicsData){
    html+='<div class="topic-card" id="tc-'+esc(t.id)+'" onclick="toggleTopic(\''+esc(t.id)+'\')">';
    html+='<div class="tc-head"><span><span class="tc-id">'+esc(t.id)+'</span><span class="tc-name">'+esc(t.name)+'</span></span>';
    html+='<span class="tc-count">'+t.count+' memories</span></div>';
    html+='<div class="tc-body" style="display:none"></div>';
    html+='</div>';
  }
  html+='</div>';
  el.innerHTML=html;
}

async function toggleTopic(topicId){
  const card=document.getElementById('tc-'+topicId);
  if(!card)return;
  const body=card.querySelector('.tc-body');
  if(body.style.display!=='none'){body.style.display='none';card.classList.remove('expanded');return;}
  card.classList.add('expanded');
  body.style.display='block';
  body.innerHTML='<div class="empty">Loading...</div>';
  try{
    const tk=localStorage.getItem('engram_token')||'';
    const ns=localStorage.getItem('engram_namespace')||'';
    const h={'Content-Type':'application/json'};
    if(tk)h['Authorization']='Bearer '+tk;
    if(ns&&ns!=='default')h['X-Namespace']=ns;
    const res=await fetch(A+'/topic?touch=false',{method:'POST',headers:h,body:JSON.stringify({ids:[topicId]})});
    const d=await res.json();
    const mems=(d[topicId]&&d[topicId].memories)||[];
    if(!mems.length){body.innerHTML='<div class="empty">No memories in this topic</div>';return;}
    const mkTags=m=>(m.tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join(' ');
    body.innerHTML='<div class="ml">'+mems.map(m=>{
      const kind='<span class="kind-badge kind-'+(m.kind||'semantic')+'">'+esc(m.kind||'semantic')+'</span>';
      return '<div class="mc" style="cursor:default"><div class="ct">'+esc(m.content||'')+'</div>'+
        '<div class="meta"><span class="lb l'+(m.layer||1)+'">'+ln(m.layer||1)+'</span> '+kind+
        '<span>imp '+(m.importance||0).toFixed(2)+'</span> <span>'+ago(m.created_at)+'</span> '+mkTags(m)+'</div></div>';
    }).join('')+'</div>';
  }catch(e){body.innerHTML='<div class="empty">Error: '+esc(e.message)+'</div>';}
}

async function doSearch(){
  const q=document.getElementById('sQ').value.trim();if(!q)return;
  document.getElementById('sRes').innerHTML='<div class="empty">Searching...</div>';
  const t0=performance.now();
  const d=await api('/search?q='+encodeURIComponent(q));
  const ms=(performance.now()-t0).toFixed(0);
  const r=Array.isArray(d)?d:(d.memories||[]);
  document.getElementById('sRes').innerHTML=`
    <div style="font-size:11px;color:#555;margin:10px 0 6px">${r.length} results, ${ms}ms (FTS)</div>
    <div class="ml">${r.length?r.map(m=>card(m,m.relevance)).join(''):'<div class="empty">No results</div>'}</div>`;
  bindCards();
}

function renderCreate(){
  document.getElementById('main').innerHTML=`
    <h2>Create</h2>
    <div class="card" style="max-width:540px">
      <textarea id="ccontent" rows="4" placeholder="Content..." style="width:100%;margin-bottom:8px"></textarea>
      <input type="text" id="ctags" placeholder="Tags (comma-separated)" style="width:100%;margin-bottom:8px">
      <input type="text" id="csource" placeholder="Source" style="width:100%;margin-bottom:8px">
      <div class="row" style="margin:0 0 8px">
        <label>Kind <select id="ckind"><option>semantic</option><option>episodic</option><option>procedural</option></select></label>
        <label>Imp <input type="range" id="cimp" min="0" max="1" step=".05" value="0.5" oninput="document.getElementById('cimpv').textContent=this.value"><span id="cimpv">0.5</span></label>
        <label><input type="checkbox" id="csync"> sync embed</label>
      </div>
      <button class="btn" onclick="doCreate()">Create</button>
    </div>`;
}

async function doCreate(){
  const content=document.getElementById('ccontent').value.trim();if(!content)return;
  const tags=document.getElementById('ctags').value.split(',').map(t=>t.trim()).filter(Boolean);
  const source=document.getElementById('csource').value.trim();
  const kind=document.getElementById('ckind').value;
  const importance=parseFloat(document.getElementById('cimp').value);
  const sync_embed=document.getElementById('csync').checked;
  const body={content,kind,importance};
  if(tags.length)body.tags=tags;if(source)body.source=source;if(sync_embed)body.sync_embed=true;
  try{
    const m=await api('/memories',{method:'POST',body:JSON.stringify(body)});
    toast('Created '+m.id.slice(0,8));document.getElementById('ccontent').value='';
    loadMem();loadStats();
  }catch(e){toast(e.message,false);}
}

function renderFacts(){
  document.getElementById('main').innerHTML=`
    <h2>Facts</h2>
    <div class="card">
      <div class="row" style="margin:0">
        <input type="text" id="fSubj" placeholder="Subject" style="width:120px">
        <input type="text" id="fPred" placeholder="Predicate" style="width:120px">
        <input type="text" id="fObj" placeholder="Object" style="width:120px">
        <button class="btn" onclick="queryFacts()">Query</button>
        <button class="btn btn-ghost" onclick="loadAllFacts()">All</button>
        <button class="btn btn-ghost" onclick="loadConflicts()">Conflicts</button>
      </div>
    </div>
    <div id="fRes"></div>`;
  loadAllFacts();
}

async function loadAllFacts(){
  const d=await api('/facts/all');renderFactTable(d.facts||d||[],'All');
}
async function queryFacts(){
  const s=document.getElementById('fSubj').value.trim(),p=document.getElementById('fPred').value.trim(),o=document.getElementById('fObj').value.trim();
  let qs=[];if(s)qs.push('subject='+encodeURIComponent(s));if(p)qs.push('predicate='+encodeURIComponent(p));if(o)qs.push('object='+encodeURIComponent(o));
  const d=await api('/facts?'+qs.join('&'));renderFactTable(d.facts||d||[],'Results');
}
async function loadConflicts(){
  const d=await api('/facts/conflicts');const c=d.conflicts||d||[];
  if(!c.length){document.getElementById('fRes').innerHTML='<div class="empty">No conflicts</div>';return;}
  let h='';
  for(const g of c){
    h+=`<div class="card" style="border-color:#c0392b44"><div style="font-weight:600;margin-bottom:6px">${esc(g.subject)} → ${esc(g.predicate)}</div>`;
    for(const f of(g.facts||[]))h+=`<div style="display:flex;justify-content:space-between;padding:2px 0"><span>${esc(f.object)} <span style="color:#555;font-size:10px">${new Date(f.created_at).toLocaleDateString()}</span></span><button class="btn btn-danger" style="padding:1px 6px;font-size:10px" onclick="delFact('${f.id}')">×</button></div>`;
    h+='</div>';
  }
  document.getElementById('fRes').innerHTML=h;
}
function renderFactTable(facts,title){
  if(!facts.length){document.getElementById('fRes').innerHTML='<div class="empty">No facts</div>';return;}
  document.getElementById('fRes').innerHTML=`
    <div style="font-size:11px;color:#555;margin:10px 0 6px">${title}: ${facts.length}</div>
    <table class="ftable"><thead><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Created</th><th></th></tr></thead>
    <tbody>${facts.map(f=>`<tr><td>${esc(f.subject)}</td><td>${esc(f.predicate)}</td><td>${esc(f.object)}</td>
      <td style="font-size:10px;color:#555">${new Date(f.created_at).toLocaleDateString()}</td>
      <td><button class="btn btn-danger" style="padding:1px 6px;font-size:9px" onclick="delFact('${f.id}')">×</button></td></tr>`).join('')}</tbody></table>`;
}
async function delFact(id){if(!confirm('Delete?'))return;await api('/facts/'+id,{method:'DELETE'});toast('Deleted');loadAllFacts();}

function renderProxy(){
  const p=health.proxy||{},mems=allMem.filter(m=>m.source==='proxy');
  document.getElementById('main').innerHTML=`
    <h2>Proxy</h2>
    <div class="row" style="align-items:center;gap:8px;margin-bottom:12px">
      <span class="dot ${p.enabled?'on':'off'}"></span> ${p.enabled?'Active':'Inactive'}
    </div>
    <div class="stats">
      <div class="sc"><div class="v">${p.requests||0}</div><div class="l">Requests</div></div>
      <div class="sc"><div class="v">${p.buffered_turns||0}</div><div class="l">Buffered</div></div>
      <div class="sc"><div class="v">${p.extracted||0}</div><div class="l">Extracted</div></div>
      <div class="sc"><div class="v">${mems.length}</div><div class="l">Memories</div></div>
    </div>
    <div class="row"><button class="btn btn-ghost" onclick="doAction('/proxy/flush','POST')">Flush</button></div>
    <div class="ml">${mems.length?mems.map(m=>card(m)).join(''):'<div class="empty">No proxy memories</div>'}</div>`;
  bindCards();
}

function renderHealth(){
  const h=health,up=h.uptime_secs||0,c=h.embed_cache||{},p=h.proxy||{};
  const hr=c.hits+c.misses>0?((c.hits/(c.hits+c.misses))*100).toFixed(1):'—';
  document.getElementById('main').innerHTML=`
    <h2>Health</h2>
    <div class="stats">
      <div class="sc"><div class="v">${h.version||'?'}</div><div class="l">Version</div></div>
      <div class="sc"><div class="v">${Math.floor(up/3600)}h${Math.floor(up%3600/60)}m</div><div class="l">Uptime</div></div>
      <div class="sc"><div class="v">${h.rss_kb?Math.round(h.rss_kb/1024):'?'}</div><div class="l">RSS MB</div></div>
      <div class="sc"><div class="v">${h.db_size_mb||'?'}</div><div class="l">DB MB</div></div>
    </div>
    <div class="card"><b style="font-size:12px">Cache</b>
      <div class="stats" style="margin:8px 0 0">
        <div class="sc"><div class="v">${c.hits||0}</div><div class="l">Hits</div></div>
        <div class="sc"><div class="v">${c.misses||0}</div><div class="l">Misses</div></div>
        <div class="sc"><div class="v">${hr}%</div><div class="l">Hit rate</div></div>
        <div class="sc"><div class="v">${c.size||0}/${c.capacity||0}</div><div class="l">Size</div></div>
      </div>
    </div>
    <div class="card"><b style="font-size:12px">Proxy</b>
      <div class="stats" style="margin:8px 0 0">
        <div class="sc"><div class="v"><span class="dot ${p.enabled?'on':'off'}"></span></div><div class="l">Status</div></div>
        <div class="sc"><div class="v">${p.requests||0}</div><div class="l">Requests</div></div>
        <div class="sc"><div class="v">${p.extracted||0}</div><div class="l">Extracted</div></div>
        <div class="sc"><div class="v">${p.buffered_turns||0}</div><div class="l">Buffered</div></div>
      </div>
    </div>
    ${stats.by_kind?`<div class="card"><b style="font-size:12px">By kind</b>
      <div class="stats" style="margin:8px 0 0">
        ${Object.entries(stats.by_kind).map(([k,v])=>`<div class="sc"><div class="v">${v}</div><div class="l">${k}</div></div>`).join('')}
      </div>
    </div>`:''}
    ${h.integrity?`<div class="card"><b style="font-size:12px">Integrity</b>
      <div class="stats" style="margin:8px 0 0">
        <div class="sc"><div class="v"><span class="dot ${h.integrity.ok?'on':'off'}"></span></div><div class="l">Status</div></div>
        <div class="sc"><div class="v">${h.integrity.missing_embedding||0}</div><div class="l">Missing embed</div></div>
        <div class="sc"><div class="v">${h.integrity.missing_fts||0}</div><div class="l">Missing FTS</div></div>
        <div class="sc"><div class="v">${h.integrity.orphan_fts||0}</div><div class="l">Orphan FTS</div></div>
      </div>
    </div>`:''}`;
}

let trashPage=0,trashTotal=0;
const TPP=30;

async function loadTrashCount(){
  try{
    const d=await api('/trash?limit=1');
    const el=document.getElementById('c-trash');
    if(el) el.textContent=d.total||d.count||0;
  }catch(_){}
}

function renderTrash(){
  document.getElementById('main').innerHTML=`
    <h2>Trash</h2>
    <div class="row" style="margin-bottom:12px;gap:8px">
      <button class="btn" onclick="loadTrashPage(0)">Refresh</button>
      <button class="btn" style="background:#7f1d1d" onclick="purgeTrash()">Purge All</button>
    </div>
    <div id="trashList"><div class="empty">Loading...</div></div>
    <div id="trashPager"></div>`;
  loadTrashPage(0);
}

async function loadTrashPage(p){
  trashPage=Math.max(0,p);
  const offset=trashPage*TPP;
  const el=document.getElementById('trashList');
  const pgEl=document.getElementById('trashPager');
  try{
    const d=await api('/trash?limit='+TPP+'&offset='+offset);
    const items=d.items||[];
    trashTotal=d.total||items.length;
    if(!items.length&&trashTotal===0){el.innerHTML='<div class="empty">Trash is empty</div>';if(pgEl)pgEl.innerHTML='';return;}
    el.innerHTML='<div class="ml">'+items.map(t=>{
      const tags=(t.tags||[]).map(tg=>'<span class="tag">'+esc(tg)+'</span>').join(' ');
      const kind='<span class="kind-badge kind-'+(t.kind||'semantic')+'">'+esc(t.kind||'semantic')+'</span>';
      return '<div class="mc" style="opacity:0.85">'+
        '<div class="ct">'+esc((t.content||'').slice(0,280))+(t.content&&t.content.length>280?'…':'')+'</div>'+
        '<div class="meta">'+
          '<span class="lb l'+t.layer+'">'+ ({1:'Buffer',2:'Working',3:'Core'}[t.layer]||'?')+'</span>'+
          kind+
          '<span>imp '+(t.importance||0).toFixed(2)+'</span>'+
          '<span>'+ago(t.created_at)+'</span>'+
          '<span style="color:#c0392b">deleted '+ago(t.deleted_at)+'</span>'+
          tags+
        '</div>'+
        '<div style="margin-top:6px">'+
          '<button class="btn btn-ghost" style="font-size:11px" onclick="event.stopPropagation();restoreTrash(\''+t.id+'\')">Restore</button>'+
        '</div>'+
      '</div>';
    }).join('')+'</div>';
    const pages=Math.ceil(trashTotal/TPP);
    if(pgEl){
      if(pages>1){
        pgEl.innerHTML='<div class="pager">'+
          '<button onclick="loadTrashPage('+(trashPage-1)+')" '+(trashPage===0?'disabled':'')+'>Prev</button>'+
          '<span>'+(trashPage+1)+'/'+pages+' ('+trashTotal+')</span>'+
          '<button onclick="loadTrashPage('+(trashPage+1)+')" '+(trashPage>=pages-1?'disabled':'')+'>Next</button>'+
        '</div>';
      }else{
        pgEl.innerHTML='<div style="text-align:center;font-size:11px;color:#555;margin-top:10px">'+trashTotal+' items</div>';
      }
    }
  }catch(e){el.innerHTML='<div class="empty">'+esc(e.message)+'</div>';}
}

async function restoreTrash(id){
  try{
    await api('/trash/'+id+'/restore',{method:'POST'});
    toast('Restored');
    await refresh();
    loadTrashPage(trashPage);
    loadTrashCount();
  }catch(e){toast(e.message,false);}
}

async function purgeTrash(){
  if(!confirm('Permanently delete all trash? This cannot be undone.'))return;
  try{
    const d=await api('/trash',{method:'DELETE'});
    toast('Purged '+(d.purged||0)+' items');
    loadTrashPage(0);
    loadTrashCount();
  }catch(e){toast(e.message,false);}
}

function renderTools(){
  document.getElementById('main').innerHTML=`
    <h2>Tools</h2>
    <div class="card" style="max-width:540px">
      <div class="row" style="margin-bottom:12px">
        <button class="btn btn-ghost" onclick="doAction('/vacuum','POST')">Vacuum</button>
        <button class="btn btn-ghost" onclick="doAction('/audit','POST','{}')">Audit</button>
        <button class="btn btn-ghost" onclick="doAction('/repair?force=true','POST')">Full rebuild</button>
      </div>
      <b style="font-size:12px">Extract from text</b>
      <textarea id="extText" rows="4" placeholder="Paste text..." style="width:100%;margin:6px 0"></textarea>
      <button class="btn" onclick="doExtract()" style="margin-bottom:12px">Extract</button>
      <div id="extRes"></div>
      <b style="font-size:12px">Import / Export</b>
      <div class="row" style="margin-top:6px">
        <button class="btn btn-ghost" onclick="doExport(false)">Export</button>
        <button class="btn btn-ghost" onclick="doExport(true)">Export + embeddings</button>
        <label class="btn btn-ghost" style="cursor:pointer">Import<input type="file" accept=".json" style="display:none" onchange="doImport(this)"></label>
      </div>
      <div id="impRes"></div>
    </div>`;
}

async function doExtract(){
  const text=document.getElementById('extText').value.trim();if(!text)return;
  document.getElementById('extRes').innerHTML='<div class="empty">Extracting...</div>';
  try{
    const d=await api('/extract',{method:'POST',body:JSON.stringify({text})});
    const mems=d.memories||d||[];
    document.getElementById('extRes').innerHTML=`<div style="font-size:11px;color:#555;margin:6px 0">Extracted ${mems.length}</div><div class="ml">${mems.map(m=>card(m)).join('')}</div>`;
    toast('Extracted '+mems.length);await refresh();
  }catch(e){toast(e.message,false);}
}
async function doExport(embed){
  const tk=localStorage.getItem('engram_token')||'';
  const ns=localStorage.getItem('engram_namespace')||'';
  const hdrs={};
  if(tk)hdrs['Authorization']='Bearer '+tk;
  if(ns&&ns!=='default')hdrs['X-Namespace']=ns;
  const res=await fetch(A+'/export'+(embed?'?embed=true':''),{headers:hdrs});
  const blob=await res.blob();const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='engram-export.json';a.click();toast('Downloaded');
}
async function doImport(input){
  const file=input.files[0];if(!file)return;const text=await file.text();
  try{const d=await api('/import',{method:'POST',body:text});
    toast('Imported '+(d.imported||0));
    document.getElementById('impRes').innerHTML=`<div style="color:#4caf50;font-size:12px;margin-top:6px">Imported ${d.imported||0}, skipped ${d.skipped||0}</div>`;
    await refresh();
  }catch(e){toast(e.message,false);}
}
async function doAction(path,method,body){
  toast(path.slice(1)+': running...');
  try{const opts={method};if(body)opts.body=typeof body==='string'?body:JSON.stringify(body);
    const d=await api(path,opts);
    const msg=typeof d==='object'&&d.summary?d.summary:(typeof d==='string'?d:JSON.stringify(d)).slice(0,120);
    toast(path.slice(1)+': '+msg);await refresh();
  }catch(e){toast(e.message,false);}
}

async function refresh(){await Promise.all([loadStats(),loadMem(),loadTrashCount()]);switchView(curView);}
function renderTests(){
  document.getElementById('main').innerHTML=`
    <h2>Tests</h2>
    <div class="row" style="margin-bottom:12px">
      <button class="btn" id="runAllBtn" onclick="runAllTests()">Run All</button>
      <span id="testSummary" style="font-size:11px;color:#555"></span>
    </div>
    <div id="testList" class="ml"></div>`;
  renderTestCards();
}

const testDefs=[
  {name:'Health check',fn:testHealth},
  {name:'Health data completeness',fn:testHealthFields},
  {name:'Stats',fn:testStats},
  {name:'Memory CRUD',fn:testMemoryCrud},
  {name:'Recall',fn:testRecall},
  {name:'Search',fn:testSearch},
  {name:'Facts CRUD',fn:testFactsCrud},
  {name:'Sort order',fn:testSortOrder},
];
let testResults={};

function renderTestCards(){
  const el=document.getElementById('testList');
  if(!el)return;
  el.innerHTML=testDefs.map((t,i)=>{
    const r=testResults[i];
    let status='<span style="color:#555">—</span>';
    let timing='';
    let detail='';
    if(r){
      if(r.running){status='<span style="color:#d4a44a">running...</span>';}
      else if(r.pass){status='<span style="color:#4caf50">PASS</span>';timing=r.ms+'ms';}
      else{status='<span style="color:#c0392b">FAIL</span>';timing=r.ms+'ms';detail=`<div style="color:#c0392b;font-size:11px;margin-top:4px">${esc(r.error)}</div>`;}
    }
    return`<div class="card" style="padding:10px 14px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:12px;color:#ccc">${esc(t.name)}</span>
        <span style="font-size:11px">${status} <span style="color:#555;margin-left:6px">${timing}</span></span>
      </div>${detail}</div>`;
  }).join('');
}

async function runAllTests(){
  const btn=document.getElementById('runAllBtn');
  btn.disabled=true;btn.textContent='Running...';
  testResults={};
  let passed=0,failed=0;
  for(let i=0;i<testDefs.length;i++){
    testResults[i]={running:true};renderTestCards();
    const t0=performance.now();
    try{
      await testDefs[i].fn();
      const ms=Math.round(performance.now()-t0);
      testResults[i]={pass:true,ms};passed++;
    }catch(e){
      const ms=Math.round(performance.now()-t0);
      testResults[i]={pass:false,ms,error:e.message||String(e)};failed++;
    }
    renderTestCards();
  }
  btn.disabled=false;btn.textContent='Run All';
  document.getElementById('testSummary').textContent=passed+' passed, '+failed+' failed';
}

function assert(cond,msg){if(!cond)throw new Error(msg||'assertion failed');}

async function testHealth(){
  const r=await api('/health');
  assert(r.version,'missing version field');
}

async function testHealthFields(){
  const r=await api('/health');
  assert(r.db_size_mb!==undefined,'missing db_size_mb');
  assert(r.rss_kb!==undefined,'missing rss_kb');
  assert(r.embed_cache,'missing embed_cache');
  assert(r.embed_cache.hits!==undefined,'missing embed_cache.hits');
  assert(r.embed_cache.misses!==undefined,'missing embed_cache.misses');
}

async function testStats(){
  const r=await api('/stats');
  assert(r.total!==undefined,'missing total');
  assert(r.buffer!==undefined,'missing buffer');
  assert(r.working!==undefined,'missing working');
  assert(r.core!==undefined,'missing core');
}

async function testMemoryCrud(){
  // create
  const m=await api('/memories',{method:'POST',body:JSON.stringify({content:'__test_crud_'+Date.now(),tags:['_test']})});
  assert(m.id,'create: no id returned');
  try{
    // read
    const got=await api('/memories/'+m.id);
    assert(got.id===m.id,'read: id mismatch');
    // update
    const updated='__test_updated_'+Date.now();
    await api('/memories/'+m.id,{method:'PATCH',body:JSON.stringify({content:updated})});
    const got2=await api('/memories/'+m.id);
    assert(got2.content===updated,'update: content not changed');
    // delete
    await api('/memories/'+m.id,{method:'DELETE'});
    // verify 404
    try{
      const tk=localStorage.getItem('engram_token')||'';
      const ns=localStorage.getItem('engram_namespace')||'';
      const h={'Content-Type':'application/json'};
      if(tk)h['Authorization']='Bearer '+tk;
      if(ns&&ns!=='default')h['X-Namespace']=ns;
      const res=await fetch(A+'/memories/'+m.id,{headers:h});
      assert(res.status===404,'delete: expected 404 got '+res.status);
    }catch(e){if(e.message.startsWith('delete:'))throw e;}
  }catch(e){
    // cleanup on failure
    try{await api('/memories/'+m.id,{method:'DELETE'});}catch(_){}
    throw e;
  }
}

async function testRecall(){
  const r=await api('/recall',{method:'POST',body:JSON.stringify({query:'test',limit:5})});
  assert(Array.isArray(r.memories),'response missing memories array');
}

async function testSearch(){
  const r=await api('/search?q=test');
  // just verify we get an array response
  const arr=Array.isArray(r)?r:(r.memories||r);
  assert(Array.isArray(arr)||typeof r==='object','unexpected search response');
}

async function testFactsCrud(){
  const subj='_test_'+Date.now();
  // create — facts API expects {facts: [...]}
  const f=await api('/facts',{method:'POST',body:JSON.stringify({facts:[{subject:subj,predicate:'is',object:'a test'}]})});
  const fact=(f.facts&&f.facts[0]);
  const fid=fact?fact.id:null;
  const mid=fact?fact.memory_id:null;
  assert(fid,'create: no id');
  try{
    // query
    const q=await api('/facts?entity='+encodeURIComponent(subj));
    const facts=q.facts||q||[];
    assert(facts.length>0,'query: no facts found');
    // delete fact + cleanup auto-created memory
    await api('/facts/'+fid,{method:'DELETE'});
    if(mid) try{await api('/memories/'+mid,{method:'DELETE'});}catch(_){}
  }catch(e){
    try{await api('/facts/'+fid,{method:'DELETE'});}catch(_){}
    if(mid) try{await api('/memories/'+mid,{method:'DELETE'});}catch(_){}
    throw e;
  }
}

async function testSortOrder(){
  const d=await api('/memories?limit=20');
  const mems=Array.isArray(d)?d:(d.memories||[]);
  if(mems.length<2)return; // can't verify with fewer than 2
  for(let i=1;i<mems.length;i++){
    assert(mems[i-1].created_at>=mems[i].created_at,
      'not sorted: index '+(i-1)+' ('+mems[i-1].created_at+') < index '+i+' ('+mems[i].created_at+')');
  }
}

/* --- LLM Usage page --- */
let llmRefreshTimer=null;

const COMP_DESC={
  gate:'Working→Core promotion decisions',
  audit:'Topic distillation — condenses bloated topics',
  distill:'Topic distillation — condenses bloated topics',
  reconcile:'Detects same-topic memories, decides update vs keep both',
  merge:'Merges near-duplicate memories into one',
  expand:'Expands search queries for better recall',
  query_expand:'Expands search queries for better recall',
  triage:'Evaluates buffer memories for promotion',
  proxy_extract:'Extracts memories from proxy conversations',
  core_summary:'Generates Core memory summaries for resume',
  naming:'Names topic tree clusters via LLM',
  embed:'Embedding operation',
  embed_batch:'Batch embedding operation',
  embed_queue:'Queued embedding operation',
  embed_sync:'Sync embedding operation',
  recall_embed:'Recall query embedding',
  merge_embed:'Merge result embedding',
  dedup_embed:'Dedup check embedding',
  insert_merge:'Dedup check on insert',
  distill:'Distills session notes into project context',
  facts_extract:'Extracts facts from memories',
  extract:'Extracts memories from text',
};

function fmtTokens(n){
  if(n==null)return '0';
  if(n>=1e6)return (n/1e6).toFixed(1)+'M';
  if(n>=1e3)return (n/1e3).toFixed(1)+'K';
  return String(n);
}

function renderLlmUsage(){
  document.getElementById('main').innerHTML=`
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <h2 style="margin:0">LLM Usage</h2>
      <button class="btn btn-danger" style="font-size:11px;padding:3px 10px" onclick="clearLlmUsage()">Clear</button>
    </div>
    <div id="llmContent"><div class="empty">Loading...</div></div>`;
  loadLlmUsage();
  // auto-refresh every 30s
  if(llmRefreshTimer)clearInterval(llmRefreshTimer);
  llmRefreshTimer=setInterval(()=>{if(curView==='llm-usage')loadLlmUsage();},30000);
}

async function clearLlmUsage(){
  if(!confirm('Clear all LLM usage data?'))return;
  try{
    await api('/llm-usage',{method:'DELETE'});
    toast('LLM usage cleared');
    loadLlmUsage();
  }catch(e){
    if(e.message&&(e.message.includes('404')||e.message.includes('405')))
      alert('DELETE /llm-usage not implemented on this server version');
    else toast(e.message,false);
  }
}

async function loadLlmUsage(){
  try{
    const d=await api('/llm-usage');
    const s=d.summary||{};
    const daily=d.daily||[];

    // Summary cards
    let html=`<div class="stats">
      <div class="sc"><div class="v">${fmtTokens(s.total_calls)}</div><div class="l">Total Calls</div></div>
      <div class="sc buf"><div class="v">${fmtTokens(s.total_input)}</div><div class="l">Input Tokens</div></div>
      <div class="sc wrk"><div class="v">${fmtTokens(s.total_output)}</div><div class="l">Output Tokens</div></div>
      <div class="sc cor"><div class="v">${fmtTokens(s.today_calls)}</div><div class="l">Today</div></div>
    </div>`;

    // By Component table
    const byComp=s.by_component||{};
    const compEntries=Object.entries(byComp).sort((a,b)=>(b[1].calls||0)-(a[1].calls||0));
    if(compEntries.length){
      html+=`<div class="card"><b style="font-size:12px">By Component</b>
        <table class="ftable" style="margin-top:8px"><thead><tr><th>Component</th><th>Calls</th><th>Input</th><th>Output</th><th>Cached</th></tr></thead><tbody>`;
      for(const[name,v]of compEntries){
        const desc=COMP_DESC[name]||'';
        html+=`<tr><td>${esc(name)}${desc?'<div style="font-size:10px;color:#666;font-weight:400;line-height:1.3">'+esc(desc)+'</div>':''}</td><td>${v.calls||0}</td><td>${fmtTokens(v.input_tokens)}</td><td>${fmtTokens(v.output_tokens)}</td><td>${fmtTokens(v.cached_tokens)}</td></tr>`;
      }
      html+=`</tbody></table></div>`;
    }

    // By Model table
    const byModel=s.by_model||{};
    const modelEntries=Object.entries(byModel).sort((a,b)=>(b[1].calls||0)-(a[1].calls||0));
    if(modelEntries.length){
      html+=`<div class="card"><b style="font-size:12px">By Model</b>
        <table class="ftable" style="margin-top:8px"><thead><tr><th>Model</th><th>Calls</th><th>Input</th><th>Output</th><th>Cached</th></tr></thead><tbody>`;
      for(const[name,v]of modelEntries){
        html+=`<tr><td>${esc(name)}</td><td>${v.calls||0}</td><td>${fmtTokens(v.input_tokens)}</td><td>${fmtTokens(v.output_tokens)}</td><td>${fmtTokens(v.cached_tokens)}</td></tr>`;
      }
      html+=`</tbody></table></div>`;
    }

    // Daily table — most recent first
    if(daily.length){
      const sorted=[...daily].sort((a,b)=>(b.date||'').localeCompare(a.date||''));
      html+=`<div class="card"><b style="font-size:12px">Daily Breakdown</b>
        <table class="ftable" style="margin-top:8px"><thead><tr><th>Date</th><th>Component</th><th>Model</th><th>Calls</th><th>Input</th><th>Output</th><th>Cached</th><th>Avg Latency</th></tr></thead><tbody>`;
      for(const r of sorted){
        html+=`<tr><td>${esc(r.date||'')}</td><td>${esc(r.component||'')}</td><td>${esc(r.model||'')}</td><td>${r.calls||0}</td><td>${fmtTokens(r.input_tokens)}</td><td>${fmtTokens(r.output_tokens)}</td><td>${fmtTokens(r.cached_tokens)}</td><td>${r.avg_duration_ms?r.avg_duration_ms+'ms':'—'}</td></tr>`;
      }
      html+=`</tbody></table></div>`;
    }

    document.getElementById('llmContent').innerHTML=html;
  }catch(e){
    document.getElementById('llmContent').innerHTML=`<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

function switchView(v){
  curView=v;page=0;
  // clear LLM auto-refresh when navigating away
  if(v!=='llm-usage'&&llmRefreshTimer){clearInterval(llmRefreshTimer);llmRefreshTimer=null;}
  document.querySelectorAll('.nav li').forEach(li=>li.classList.toggle('active',li.dataset.v===v));
  if(v==='recall')renderRecall();else if(v==='search')renderSearch();else if(v==='resume')renderResume();else if(v==='topics')renderTopics();else if(v==='create')renderCreate();
  else if(v==='facts')renderFacts();else if(v==='proxy')renderProxy();else if(v==='health')renderHealth();
  else if(v==='trash')renderTrash();else if(v==='tools')renderTools();else if(v==='llm-usage')renderLlmUsage();else if(v==='tests')renderTests();else renderList(getMem());
}

document.querySelectorAll('.nav').forEach(n=>{n.onclick=e=>{const li=e.target.closest('li');if(li)switchView(li.dataset.v);};});
document.onkeydown=e=>{if(e.key==='Escape')closeDetail();if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();switchView('search');setTimeout(()=>{const el=document.getElementById('sQ');if(el)el.focus();},50);}};
setInterval(()=>Promise.all([loadStats(),loadMem()]),30000);
refresh();
