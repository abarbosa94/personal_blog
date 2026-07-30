const $ = id => document.getElementById(id);
const codes = {
  api_error:"API error", not_found:"Not found", ambiguous_match:"Ambiguous match",
  wrong_work:"Wrong OpenAlex work", missing_affiliation:"Missing affiliation",
  incomplete_affiliations:"Incomplete affiliations", wrong_affiliation:"Wrong affiliation",
  ror_mismatch:"ROR mismatch", other:"Other"
};
const state = {items:[], filtered:[], current:null, saving:false};

function selectedCodes(){
  return [...document.querySelectorAll("[data-code]:checked")].map(x=>x.dataset.code);
}
function current(){ return state.items.find(x=>x.review_id===state.current); }
function applyFilters(){
  const status=$("statusFilter").value, venue=$("venueFilter").value, signal=$("signalFilter").value;
  state.filtered=state.items.filter(x =>
    (status==="all" || (status==="open" ? ["needs_review","defer"].includes(x.review_status) : x.review_status===status)) &&
    (venue==="all" || x.venue===venue) && (signal==="all" || x.system_signal===signal));
  if(!state.filtered.some(x=>x.review_id===state.current)) state.current=state.filtered[0]?.review_id||null;
  render();
}
function render(){
  const item=current(), meta=state.meta;
  $("progressText").textContent=`${meta.completed} completed · ${meta.deferred} deferred · ${meta.total} total`;
  $("progressBar").style.width=`${meta.total ? 100*meta.completed/meta.total : 0}%`;
  $("empty").hidden=!!item; $("review").hidden=!item;
  if(!item) return;
  $("venueBadge").textContent=item.venue.toUpperCase(); $("trackBadge").textContent=item.track;
  $("priorityBadge").textContent=`${item.priority} priority`; $("statusBadge").textContent=item.review_status.replace("_"," ");
  $("title").textContent=item.title; $("authors").textContent=item.authors.split(" | ").join(", ");
  setLink("officialLink",item.official_url); setLink("pdfLink",item.pdf_url); setLink("openalexLink",item.openalex_id);
  $("method").textContent=item.match_method||"none"; $("signal").textContent=item.system_signal;
  $("reason").textContent=item.priority_reason; $("paperId").textContent=item.paper_id;
  const aff=JSON.parse(item.affiliations_json||"[]");
  $("affiliations").innerHTML=aff.length ? aff.map(a=>`<div class="affiliation"><strong>${escapeHtml(a.institution_name||"Unnamed")}</strong><span>${escapeHtml(a.country_code||"country unresolved")} · ${escapeHtml(a.institution_id||"no ROR/OpenAlex ID")}</span></div>`).join("") : `<p class="muted">No affiliations recorded.</p>`;
  document.querySelectorAll("[data-status]").forEach(x=>x.classList.toggle("selected",x.dataset.status===item.review_status));
  document.querySelectorAll("[name=confidence]").forEach(x=>x.checked=x.value===item.confidence);
  const chosen=new Set((item.failure_codes||"").split("|").filter(Boolean));
  document.querySelectorAll("[data-code]").forEach(x=>x.checked=chosen.has(x.dataset.code));
  $("note").value=item.review_note; $("regression").checked=item.add_to_regression==="true";
  $("failureFieldset").hidden=item.review_status!=="fail";
  $("validation").hidden=true;
}
function setLink(id,url){ const node=$(id); node.hidden=!url; if(url) node.href=url; }
function escapeHtml(value){ const node=document.createElement("span"); node.textContent=value; return node.innerHTML; }
function go(delta){
  if(!state.filtered.length)return;
  let i=state.filtered.findIndex(x=>x.review_id===state.current);
  state.current=state.filtered[(i+delta+state.filtered.length)%state.filtered.length].review_id; render(); scrollTo(0,0);
}
function choose(status){
  const item=current(); if(!item)return;
  item.review_status=status;
  if(status==="pass") item.failure_codes="";
  $("failureFieldset").hidden=status!=="fail";
  render();
}
async function save(){
  const item=current(); if(!item||state.saving)return;
  const confidence=document.querySelector("[name=confidence]:checked")?.value||"";
  const changes={review_status:item.review_status,confidence,
    failure_codes:item.review_status==="fail"?selectedCodes().join("|"):"",
    review_note:$("note").value,add_to_regression:String($("regression").checked)};
  state.saving=true; $("saveStatus").textContent="Saving…"; $("validation").hidden=true;
  try{
    const response=await fetch(`/api/items/${encodeURIComponent(item.review_id)}`,{
      method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify(changes)});
    const body=await response.json(); if(!response.ok)throw new Error(body.error||"Save failed");
    Object.assign(item,body.item); $("saveStatus").textContent="Saved";
    const snapshot=await fetch("/api/items").then(x=>x.json()); state.meta=snapshot.meta;
    applyFilters(); if(current()?.review_id===item.review_id)go(1);
  }catch(error){$("validation").textContent=error.message;$("validation").hidden=false;$("saveStatus").textContent="Not saved";}
  finally{state.saving=false;}
}
async function init(){
  $("failureCodes").innerHTML=Object.entries(codes).map(([value,label])=>`<label><input type="checkbox" data-code="${value}"> ${label}</label>`).join("");
  const body=await fetch("/api/items").then(x=>x.json()); state.items=body.items; state.meta=body.meta;
  for(const venue of [...new Set(state.items.map(x=>x.venue))].sort()) $("venueFilter").add(new Option(venue.toUpperCase(),venue));
  for(const signal of [...new Set(state.items.map(x=>x.system_signal))].sort()) $("signalFilter").add(new Option(signal.replaceAll("_"," "),signal));
  state.current=state.items.find(x=>x.review_status==="needs_review")?.review_id||state.items[0]?.review_id;
  ["statusFilter","venueFilter","signalFilter"].forEach(id=>$(id).addEventListener("change",applyFilters));
  $("previous").onclick=()=>go(-1); $("next").onclick=()=>go(1); $("save").onclick=save;
  document.querySelectorAll("[data-status]").forEach(x=>x.onclick=()=>choose(x.dataset.status));
  document.addEventListener("keydown",event=>{if(["TEXTAREA","INPUT","SELECT"].includes(event.target.tagName))return;
    if(event.key.toLowerCase()==="p")choose("pass"); if(event.key.toLowerCase()==="f")choose("fail");
    if(event.key.toLowerCase()==="d")choose("defer"); if(event.key.toLowerCase()==="s")save();
    if(event.key==="ArrowRight")go(1); if(event.key==="ArrowLeft")go(-1);});
  applyFilters(); $("saveStatus").textContent="Ready";
}
init().catch(error=>{$("saveStatus").textContent=error.message;});
