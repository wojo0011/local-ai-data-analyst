import { describe, pearson, linearRegression, histogram, cosineSimilarity } from './lib/stats.js?v=20260817-1';
import { buildColumnDescription, inferDatasetQuestions } from './lib/schema.js?v=20260817-1';

const DUCKDB_VERSION = '1.32.0';
const DUCKDB_MODULE = `https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@${DUCKDB_VERSION}/+esm`;
const TRANSFORMERS_MODULE = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';
const WEBLLM_MODULE = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/+esm';
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

const state = { db:null, conn:null, rows:[], columns:[], numeric:[], profiles:{}, questions:{semantic:[],analyst:[]}, sourceName:'', lastQuery:[], embedder:null, llm:null, statsRequest:0 };
const $ = id => document.getElementById(id);
const safe = value => value == null ? '' : (typeof value === 'bigint' ? Number(value) : value);
const fmt = (value, digits=3) => Number.isFinite(value) ? Intl.NumberFormat('en-CA',{maximumFractionDigits:digits}).format(value) : '—';
const quote = name => `"${String(name).replaceAll('"','""')}"`;
const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

function setStatus(id,text,kind=''){const el=$(id);if(!el)return;el.textContent=text;el.dataset.kind=kind}
function renderMath(root=document.body){ if(window.renderMathInElement) window.renderMathInElement(root,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\(',right:'\\)',display:false},{left:'\\[',right:'\\]',display:true}],throwOnError:false}); }

async function initDuckDB(){
  let workerUrl='';
  try{
    setStatus('duck-status',`Loading ${DUCKDB_VERSION}…`);
    const duckdb=await import(DUCKDB_MODULE);
    const bundles=duckdb.getJsDelivrBundles();
    const bundle=await duckdb.selectBundle(bundles);
    if(!bundle?.mainWorker||!bundle?.mainModule)throw new Error('DuckDB could not select a compatible browser bundle.');
    workerUrl=URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`],{type:'text/javascript'}));
    const worker=new Worker(workerUrl);
    const logger=new duckdb.ConsoleLogger();
    state.db=new duckdb.AsyncDuckDB(logger,worker);
    await state.db.instantiate(bundle.mainModule,bundle.pthreadWorker);
    state.conn=await state.db.connect();
    const smoke=await state.conn.query('SELECT 42 AS answer');
    if(Number(safe(smoke.toArray()[0]?.answer))!==42)throw new Error('DuckDB initialized but failed its SQL smoke test.');
    setStatus('duck-status',`Ready · ${DUCKDB_VERSION}`,'ok');
  }catch(error){
    console.error('DuckDB initialization failed',error);
    try{await state.conn?.close?.()}catch{}
    try{await state.db?.terminate?.()}catch{}
    state.conn=null;state.db=null;
    setStatus('duck-status','Unavailable','error');
    setStatus('data-status',`DuckDB failed to initialize: ${error.message}`,'error');
  }finally{
    if(workerUrl)URL.revokeObjectURL(workerUrl);
  }
}

function detectWebGPU(){const ok=Boolean(navigator.gpu);setStatus('gpu-status',ok?'Available':'Not available',ok?'ok':'warn');if(!ok)setStatus('llm-status','Needs WebGPU','warn')}

async function loadCSVText(text,name='pasted.csv'){
  if(!state.db||!state.conn){setStatus('data-status','DuckDB is still initializing. Try again in a moment.','warn');return}
  if(!text.trim()){setStatus('data-status','CSV input is empty.','warn');return}
  setStatus('data-status',`Loading ${name}…`);
  try{
    const filename='analysis.csv';
    await state.db.registerFileText(filename,text);
    await state.conn.query('DROP VIEW IF EXISTS data');
    await state.conn.query(`CREATE VIEW data AS SELECT * FROM read_csv_auto('${filename}', header=true, sample_size=-1)`);
    state.sourceName=name;
    await profileDataset();
    setStatus('data-status',`${name} loaded successfully.`,'ok');
  }catch(error){console.error(error);setStatus('data-status',`Could not parse CSV: ${error.message}`,'error')}
}

async function profileDataset(){
  const schemaTable=await state.conn.query('DESCRIBE data');
  const schema=schemaTable.toArray().map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,safe(v)])));
  state.columns=schema.map(r=>({name:r.column_name ?? r.column ?? Object.values(r)[0],type:r.column_type ?? r.type ?? Object.values(r)[1]}));
  const numericPattern=/(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|DECIMAL|REAL)/i;
  state.numeric=state.columns.filter(c=>numericPattern.test(c.type)).map(c=>c.name);
  state.profiles={};
  const count=await state.conn.query('SELECT COUNT(*) AS n FROM data');
  const rowCount=Number(safe(count.toArray()[0].n));
  const missingExpressions=state.columns.map(c=>`SUM(CASE WHEN ${quote(c.name)} IS NULL THEN 1 ELSE 0 END) AS ${quote(c.name)}`).join(',');
  const missTable=state.columns.length?await state.conn.query(`SELECT ${missingExpressions} FROM data`):null;
  const misses=missTable?missTable.toArray()[0]:{};
  const totalMissing=Object.values(misses).reduce((s,v)=>s+Number(safe(v)||0),0);
  const preview=await state.conn.query('SELECT * FROM data LIMIT 50');
  state.rows=preview.toArray().map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,safe(v)])));
  $('dataset-name').textContent=state.sourceName;
  $('dataset-metrics').innerHTML=`<div class="metric"><span>Rows</span><strong>${fmt(rowCount,0)}</strong><small>observations</small></div><div class="metric"><span>Columns</span><strong>${state.columns.length}</strong><small>variables</small></div><div class="metric"><span>Numeric</span><strong>${state.numeric.length}</strong><small>quantitative</small></div><div class="metric"><span>Missing</span><strong>${fmt(totalMissing,0)}</strong><small>cells</small></div>`;
  await renderSchema(misses,rowCount); renderTable('preview-table',state.rows); populateSelectors(); renderSuggestedQuestions(); await updateStats(); renderDefaultChart();
}

async function renderSchema(misses,rowCount){
  const target=$('schema-list');target.innerHTML='';
  for(const col of state.columns){
    let cardinality='—',example='';
    try{const q=await state.conn.query(`SELECT COUNT(DISTINCT ${quote(col.name)}) AS n, MIN(CAST(${quote(col.name)} AS VARCHAR)) AS example FROM data`);const r=q.toArray()[0];cardinality=fmt(Number(safe(r.n)),0);example=safe(r.example)??''}catch{}
    const missing=Number(safe(misses[col.name])||0),pct=rowCount?missing/rowCount*100:0;
    const examples=[...new Set(state.rows.map(row=>safe(row[col.name])).filter(value=>value!==''&&value!=null).map(String))].slice(0,3);
    state.profiles[col.name]={missing,missingPercent:pct,cardinality:Number(String(cardinality).replaceAll(',','')),examples,role:state.numeric.includes(col.name)?'Numeric measure':/DATE|TIME/i.test(col.type)?'Date or time dimension':'Categorical or descriptive dimension'};
    const row=document.createElement('div');row.className='schema-item';row.innerHTML=`<code>${escapeHtml(col.name)}</code><span>${escapeHtml(col.type)}</span><span>${fmt(pct,1)}% missing</span><span>${cardinality} unique · ${escapeHtml(example)}</span>`;target.append(row);
  }
}

function renderTable(id,rows){const table=$(id);if(!rows?.length){table.innerHTML='<tbody><tr><td class="empty-cell">No rows returned.</td></tr></tbody>';return}const cols=Object.keys(rows[0]);table.innerHTML=`<thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,100).map(r=>`<tr>${cols.map(c=>`<td>${escapeHtml(safe(r[c]))}</td>`).join('')}</tr>`).join('')}</tbody>`}

function populateSelectors(){
  setSelectOptions('chart-x',state.columns.map(c=>({value:c.name,label:`${c.name} · ${c.type}`})),'No columns');
  setSelectOptions('stats-column',state.numeric.map(name=>({value:name,label:name})),'No numeric columns');
  setSelectOptions('chart-y',state.numeric.map(name=>({value:name,label:name})),'No numeric columns');
  if(state.numeric[0]){$('stats-column').value=state.numeric[0];$('chart-y').value=state.numeric[Math.min(1,state.numeric.length-1)]||state.numeric[0]}
  syncCompareOptions();
  const dateLike=state.columns.find(c=>/DATE|TIME/i.test(c.type))?.name ?? state.columns[0]?.name;
  if(dateLike)$('chart-x').value=dateLike;
}

function setSelectOptions(id,options,emptyLabel){
  const select=$(id);select.replaceChildren();
  if(!options.length){select.add(new Option(emptyLabel,''));return}
  options.forEach(option=>select.add(new Option(option.label,option.value)));
}

function syncCompareOptions(){
  const select=$('compare-column'),primary=$('stats-column').value,previous=select.value;
  const options=[{value:'',label:'No comparison · univariate'},...state.numeric.filter(name=>name!==primary).map(name=>({value:name,label:name}))];
  setSelectOptions('compare-column',options,'No comparison · univariate');
  select.value=options.some(option=>option.value===previous)?previous:'';
}

async function columnValues(name){if(!name)return[];const q=await state.conn.query(`SELECT v FROM (SELECT TRY_CAST(${quote(name)} AS DOUBLE) AS v FROM data) WHERE v IS NOT NULL AND isfinite(v) LIMIT 250000`);return q.toArray().map(r=>Number(safe(r.v))).filter(Number.isFinite)}
async function pairedValues(x,y){const q=await state.conn.query(`SELECT x,y FROM (SELECT TRY_CAST(${quote(x)} AS DOUBLE) AS x,TRY_CAST(${quote(y)} AS DOUBLE) AS y FROM data) WHERE x IS NOT NULL AND y IS NOT NULL AND isfinite(x) AND isfinite(y) LIMIT 250000`);const rows=q.toArray();return {x:rows.map(r=>Number(safe(r.x))).filter(Number.isFinite),y:rows.map(r=>Number(safe(r.y))).filter(Number.isFinite)}}

async function updateStats(){
  const request=++state.statsRequest,xName=$('stats-column').value,yName=$('compare-column').value;
  if(!xName){setStatus('stats-status','No numeric variables were detected.','warn');return}
  setStatus('stats-status',`Calculating ${xName}…`);
  try{
    const x=await columnValues(xName);if(request!==state.statsRequest)return;const d=describe(x);
    $('stat-mean').textContent=d.n?fmt(d.mean):'No numeric values';
    $('stat-sd').textContent=d.n>1?fmt(d.sd):d.n===1?'Needs 2+ values':'No numeric values';
    $('stat-median').textContent=d.n?fmt(d.median):'No numeric values';
    $('stat-outliers').textContent=d.n?`${fmt(d.outliers,0)} flagged`:'No numeric values';
    if(yName){
      const p=await pairedValues(xName,yName);if(request!==state.statsRequest)return;
      const r=pearson(p.x,p.y),reg=linearRegression(p.x,p.y);
      $('stat-correlation').textContent=Number.isFinite(r)?`r = ${fmt(r)}`:p.x.length<2?'Needs 2+ pairs':'Undefined · zero variance';
      $('stat-regression').textContent=Number.isFinite(reg.slope)?`β₁ ${fmt(reg.slope)} · R² ${fmt(reg.r2)}`:p.x.length<2?'Needs 2+ pairs':'Undefined · zero variance';
      setStatus('stats-status',`${xName}: ${fmt(d.n,0)} valid values · compared with ${yName} using ${fmt(p.x.length,0)} complete pairs.`,'ok');
    }else{
      $('stat-correlation').textContent='Comparison off';$('stat-regression').textContent='Comparison off';
      setStatus('stats-status',`${xName}: ${fmt(d.n,0)} valid numeric values · univariate analysis.`,'ok');
    }
  }catch(error){console.error(error);setStatus('stats-status',`Statistics could not be calculated: ${error.message}`,'error')}
}

function renderQuestionButtons(id,questions,targetId){
  const root=$(id);root.replaceChildren();
  if(!questions.length){root.innerHTML='<p class="empty-state">Load data to generate relevant questions.</p>';return}
  questions.forEach(question=>{const button=document.createElement('button');button.type='button';button.className='question-chip';button.textContent=question;button.addEventListener('click',()=>{$(targetId).value=question;$(targetId).focus()});root.append(button)});
}

function renderSuggestedQuestions(){
  state.questions=inferDatasetQuestions({columns:state.columns,numeric:state.numeric,profiles:state.profiles,sourceName:state.sourceName});
  renderQuestionButtons('semantic-suggestions',state.questions.semantic,'semantic-question');
  renderQuestionButtons('analyst-suggestions',state.questions.analyst,'analyst-question');
  if(!$('semantic-question').value.trim())$('semantic-question').value=state.questions.semantic[0]||'';
  if(!$('analyst-question').value.trim())$('analyst-question').value=state.questions.analyst[0]||'';
}

function isReadOnlySQL(sql){const cleaned=sql.trim().replace(/^--.*$/gm,'').trim().toUpperCase();return /^(SELECT|WITH|DESCRIBE|SUMMARIZE|EXPLAIN|SHOW)\b/.test(cleaned)&&!/(\bCOPY\b|\bEXPORT\b|\bIMPORT\b|\bINSTALL\b|\bLOAD\b|\bATTACH\b|\bDETACH\b|\bCREATE\b|\bDROP\b|\bALTER\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bCALL\b|\bPRAGMA\b)/.test(cleaned)}
async function runSQL(){const sql=$('sql-editor').value;if(!state.conn){setStatus('sql-status','DuckDB is not ready.','warn');return}if(!isReadOnlySQL(sql)){setStatus('sql-status','For privacy/safety, the lab accepts read-only analytical SQL only.','warn');return}const started=performance.now();try{const q=await state.conn.query(sql);state.lastQuery=q.toArray().map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,safe(v)])));renderTable('sql-table',state.lastQuery);$('query-time').textContent=`${Math.round(performance.now()-started)} ms`;setStatus('sql-status',`${state.lastQuery.length} row(s) shown.`,'ok')}catch(error){console.error(error);setStatus('sql-status',error.message,'error')}}

function sqlTemplate(kind){if(kind==='preview')return 'SELECT * FROM data LIMIT 25;';if(kind==='missing')return `SELECT\n  COUNT(*) AS rows\nFROM data;\n\n-- Inspect specific fields with:\n-- SELECT COUNT(*) FILTER (WHERE column_name IS NULL) FROM data;`;if(kind==='numeric'&&state.numeric.length){return `SUMMARIZE SELECT ${state.numeric.slice(0,8).map(quote).join(', ')} FROM data;`}return 'SELECT * FROM data LIMIT 25;'}

async function chartData(type,xName,yName){
  if(type==='histogram'){const vals=await columnValues(yName||xName);return histogram(vals,14)}
  if(type==='bar'){const q=await state.conn.query(`SELECT CAST(${quote(xName)} AS VARCHAR) AS x, AVG(${quote(yName)}) AS y FROM data WHERE ${quote(yName)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 30`);return q.toArray().map(r=>({x:safe(r.x),y:Number(safe(r.y))}))}
  const q=await state.conn.query(`SELECT ${quote(xName)} AS x, ${quote(yName)} AS y FROM data WHERE ${quote(xName)} IS NOT NULL AND ${quote(yName)} IS NOT NULL LIMIT 5000`);return q.toArray().map(r=>({x:safe(r.x),y:Number(safe(r.y))}));
}
async function renderDefaultChart(){if(!state.columns.length||!state.numeric.length)return;renderChart()}
async function renderChart(){const type=$('chart-type').value,xName=$('chart-x').value,yName=$('chart-y').value;if(!xName||!yName||!window.echarts)return;const data=await chartData(type,xName,yName);const chart=echarts.getInstanceByDom($('main-chart'))||echarts.init($('main-chart'));const theme=getComputedStyle(document.documentElement),text=theme.getPropertyValue('--ink').trim(),muted=theme.getPropertyValue('--ink-soft').trim(),line=theme.getPropertyValue('--line-soft').trim(),accent=theme.getPropertyValue('--accent').trim();let option={animationDuration:400,backgroundColor:'transparent',textStyle:{color:text,fontFamily:'Manrope'},grid:{left:55,right:25,top:30,bottom:55},tooltip:{trigger:'item'},xAxis:{name:xName,nameLocation:'middle',nameGap:35,axisLabel:{color:muted},axisLine:{lineStyle:{color:line}},splitLine:{lineStyle:{color:line}}},yAxis:{name:yName,axisLabel:{color:muted},axisLine:{lineStyle:{color:line}},splitLine:{lineStyle:{color:line}}},series:[]};if(type==='histogram'){option.xAxis.type='category';option.xAxis.data=data.map(b=>`${fmt(b.x0,2)}–${fmt(b.x1,2)}`);option.yAxis.name='Count';option.series=[{type:'bar',data:data.map(b=>b.count),itemStyle:{color:accent}}]}else if(type==='bar'){option.xAxis.type='category';option.xAxis.data=data.map(d=>d.x);option.series=[{type:'bar',data:data.map(d=>d.y),itemStyle:{color:accent}}]}else if(type==='line'){option.xAxis.type='category';option.xAxis.data=data.map(d=>String(d.x));option.series=[{type:'line',data:data.map(d=>d.y),showSymbol:false,lineStyle:{color:accent},itemStyle:{color:accent}}]}else{option.xAxis.type='value';option.series=[{type:'scatter',data:data.map(d=>[Number(d.x),d.y]).filter(d=>d.every(Number.isFinite)),symbolSize:7,itemStyle:{color:accent,opacity:.75}}]}chart.setOption(option,true)}

function renderPipeline(){if(!window.d3)return;const svg=d3.select('#pipeline-graph'),node=svg.node(),w=node.clientWidth||520,h=node.clientHeight||300;svg.attr('viewBox',`0 0 ${w} ${h}`).selectAll('*').remove();const nodes=[{id:'CSV',x:45,y:h/2},{id:'DuckDB',x:w*.28,y:h/2},{id:'Stats',x:w*.48,y:h*.28},{id:'Transformers',x:w*.48,y:h*.72},{id:'WebLLM',x:w*.7,y:h*.72},{id:'Charts',x:w-55,y:h/2}],links=[['CSV','DuckDB'],['DuckDB','Stats'],['DuckDB','Transformers'],['Stats','Charts'],['Transformers','WebLLM'],['WebLLM','Charts']];const byId=Object.fromEntries(nodes.map(n=>[n.id,n]));const css=getComputedStyle(document.documentElement),accent=css.getPropertyValue('--accent').trim(),muted=css.getPropertyValue('--ink-soft').trim(),paper=css.getPropertyValue('--paper').trim();svg.append('g').selectAll('line').data(links).join('line').attr('x1',d=>byId[d[0]].x).attr('y1',d=>byId[d[0]].y).attr('x2',d=>byId[d[1]].x).attr('y2',d=>byId[d[1]].y).attr('stroke',muted).attr('stroke-opacity',.45).attr('stroke-width',1.5);const g=svg.append('g').selectAll('g').data(nodes).join('g').attr('transform',d=>`translate(${d.x},${d.y})`);g.append('circle').attr('r',30).attr('fill',paper).attr('stroke',accent).attr('stroke-width',1.5);g.append('text').attr('text-anchor','middle').attr('dy','.35em').attr('fill',accent).attr('font-family','Teko').attr('font-size',d=>d.id.length>8?12:15).text(d=>d.id)}

async function loadEmbeddings(){if(state.embedder)return;setStatus('embedding-status','Loading…');try{const {pipeline,env}=await import(TRANSFORMERS_MODULE);env.allowLocalModels=false;state.embedder=await pipeline('feature-extraction',EMBEDDING_MODEL,{dtype:'q8'});setStatus('embedding-status','Ready','ok')}catch(error){console.error(error);setStatus('embedding-status','Failed','error');throw error}}
async function embedding(text){const output=await state.embedder(text,{pooling:'mean',normalize:true});return Array.from(output.data)}
async function semanticSearch(){if(!state.columns.length){$('semantic-results').innerHTML='<p class="empty-state">Load a dataset first.</p>';return}try{await loadEmbeddings();const question=$('semantic-question').value.trim()||state.questions.semantic[0]||'Which columns are most useful for explaining performance?';setStatus('embedding-status','Computing…');const q=await embedding(question);const results=[];for(const c of state.columns){const text=buildColumnDescription(c,state.profiles[c.name]);const v=await embedding(text);results.push({name:c.name,type:c.type,score:cosineSimilarity(q,v)})}results.sort((a,b)=>b.score-a.score);$('semantic-results').innerHTML=results.slice(0,6).map(r=>`<div class="semantic-hit"><div><strong>${escapeHtml(r.name)}</strong><small>${escapeHtml(r.type)}</small></div><span>${fmt(r.score,3)}</span></div>`).join('');setStatus('embedding-status','Ready','ok')}catch(error){$('semantic-results').innerHTML=`<p class="empty-state">Embedding model error: ${escapeHtml(error.message)}</p>`}}

async function loadLLM(){if(state.llm)return;if(!navigator.gpu){setStatus('llm-status','WebGPU unavailable','error');return}setStatus('llm-status','Loading…');try{const webllm=await import(WEBLLM_MODULE);const preferred=$('llm-model').value;const available=webllm.prebuiltAppConfig?.model_list?.map(m=>m.model_id)||[];let model=available.includes(preferred)?preferred:available.find(id=>/SmolLM2.*360M.*Instruct/i.test(id))||available.find(id=>/1B.*Instruct/i.test(id))||available[0];if(!model)throw new Error('No prebuilt WebLLM model was found.');$('llm-model').innerHTML=available.slice(0,30).map(id=>`<option value="${escapeHtml(id)}" ${id===model?'selected':''}>${escapeHtml(id)}</option>`).join('');state.llm=await webllm.CreateMLCEngine(model,{initProgressCallback:report=>{const p=Math.max(0,Math.min(1,report.progress||0));$('llm-progress').style.width=`${p*100}%`;setStatus('llm-status',`${Math.round(p*100)}%`)}});setStatus('llm-status','Ready','ok')}catch(error){console.error(error);setStatus('llm-status','Failed','error');$('analyst-answer').textContent=`WebLLM could not load: ${error.message}`}}

async function buildEvidenceContext(){const schema=state.columns.map(c=>buildColumnDescription(c,state.profiles[c.name])).join('\n');const summaries=[];for(const name of state.numeric.slice(0,10)){const d=describe(await columnValues(name));summaries.push(`${name}: n=${d.n}, mean=${fmt(d.mean)}, median=${fmt(d.median)}, sd=${fmt(d.sd)}, min=${fmt(d.min)}, max=${fmt(d.max)}, IQR-outliers=${d.outliers}`)}const recent=state.lastQuery.length?JSON.stringify(state.lastQuery.slice(0,15)): 'No SQL query has been run yet.';return `Dataset: ${state.sourceName}\nSchema profiles:\n${schema}\nNumeric summaries:\n${summaries.join('\n')}\nData-specific questions:\n${state.questions.analyst.join('\n')}\nRecent SQL result: ${recent}\nSmall preview: ${JSON.stringify(state.rows.slice(0,8))}`}
async function askAnalyst(){if(!state.columns.length){$('analyst-answer').textContent='Load a dataset first.';return}await loadLLM();if(!state.llm)return;const question=$('analyst-question').value.trim()||state.questions.analyst[0]||'What are the strongest signals in this dataset, and what should I query next?';$('analyst-answer').textContent='Analyzing local evidence…';try{const context=await buildEvidenceContext();const response=await state.llm.chat.completions.create({messages:[{role:'system',content:'You are a careful local data analyst. Use only the evidence provided. Never invent unavailable numbers. Separate observed evidence from hypotheses. Correlation is not causation. Recommend reproducible SQL or statistical checks for claims. Keep the response concise and educational.'},{role:'user',content:`${context}\n\nQuestion: ${question}`}],temperature:.2,max_tokens:700});$('analyst-answer').textContent=response.choices?.[0]?.message?.content||'No response returned.'}catch(error){console.error(error);$('analyst-answer').textContent=`Local analyst error: ${error.message}`}}

function toggleTheme(){const next=document.documentElement.dataset.theme==='light'?'dark':'light';document.documentElement.dataset.theme=next;try{localStorage.setItem('vis_mode',next)}catch{};renderPipeline();renderChart()}
function bindEvents(){
  $('theme-toggle').addEventListener('click',toggleTheme);$('jump-upload').addEventListener('click',()=>{$('workspace').scrollIntoView({behavior:'smooth'});$('csv-file').click()});
  $('load-demo').addEventListener('click',async()=>{const r=await fetch('./sample-data.csv');loadCSVText(await r.text(),'Synthetic retail demo')});
  $('csv-file').addEventListener('change',async e=>{const f=e.target.files?.[0];if(f)loadCSVText(await f.text(),f.name)});$('load-paste').addEventListener('click',()=>loadCSVText($('csv-paste').value,'Pasted CSV'));
  const dz=$('dropzone');['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('dragover')}));['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('dragover')}));dz.addEventListener('drop',async e=>{const f=e.dataTransfer.files?.[0];if(f)loadCSVText(await f.text(),f.name)});
  $('run-sql').addEventListener('click',runSQL);document.querySelectorAll('[data-sql-template]').forEach(b=>b.addEventListener('click',()=>{$('sql-editor').value=sqlTemplate(b.dataset.sqlTemplate)}));
  $('stats-column').addEventListener('change',()=>{syncCompareOptions();updateStats()});$('compare-column').addEventListener('change',updateStats);$('render-chart').addEventListener('click',renderChart);$('chart-type').addEventListener('change',renderChart);$('load-embeddings').addEventListener('click',loadEmbeddings);$('semantic-search').addEventListener('click',semanticSearch);$('load-llm').addEventListener('click',loadLLM);$('ask-analyst').addEventListener('click',askAnalyst);window.addEventListener('resize',()=>{renderPipeline();const c=echarts.getInstanceByDom($('main-chart'));c?.resize()});
}

async function boot(){detectWebGPU();bindEvents();renderMath();renderPipeline();await initDuckDB();window.siteLoader?.setProgress?.(88,'Preparing analyst workspace');window.siteLoader?.appReady?.()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
