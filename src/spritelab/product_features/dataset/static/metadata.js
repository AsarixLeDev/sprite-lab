(() => {
  "use strict";
  const list=document.querySelector(".metadata-pack-list");
  const status=document.getElementById("metadata-status");
  if(!list||!status)return;
  const approvalId=list.dataset.approvalId||null;
  const csrf=document.querySelector('meta[name="spritelab-csrf"]')?.content||"";
  const complete=document.getElementById("metadata-complete");
  const cancel=document.getElementById("metadata-cancel");
  const post=async(url,body)=>{const response=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrf},body:JSON.stringify(body)});const payload=await response.json();if(!response.ok)throw new Error(payload.message||payload.detail||"The pack information could not be saved.");return payload;};
  const base=(pack)=>({approval_id:approvalId,pack_id:pack.dataset.packId});
  const announce=(text)=>{status.textContent=text;};
  const syncOriginalDeclaration=(form)=>{const source=form.elements.source_type;const declaration=form.elements.original_work_declaration;if(!source||!declaration)return;const original=source.value==="my_original_work";declaration.disabled=!original;if(!original)declaration.checked=false;};
  for(const form of list.querySelectorAll(".metadata-form")){syncOriginalDeclaration(form);form.elements.source_type?.addEventListener("change",()=>syncOriginalDeclaration(form));}
  list.addEventListener("submit",async(event)=>{
    const form=event.target.closest(".metadata-form");if(!form)return;event.preventDefault();
    const pack=form.closest(".metadata-pack");const submit=form.querySelector('[type="submit"]');submit.disabled=true;
    const data=Object.fromEntries(new FormData(form).entries());
    data.original_work_declaration=form.elements.original_work_declaration.checked;
    data.permission_confirmed=form.elements.permission_confirmed.checked;
    try{const result=await post("/dataset/api/metadata/save",{...base(pack),metadata:data});pack.querySelector(".status-pill").textContent="Complete";announce(`Saved pack information. ${result.inspection.image_count} PNG file(s) remain protected by project-side metadata.`);if(!result.inspection.wizard_required){const returning=document.getElementById("metadata-return");document.getElementById("metadata-next").textContent=complete?"All pack declarations are complete. Choose Complete and continue to resume the command.":"All pack declarations are complete. Continue to the dataset build.";if(complete)complete.disabled=false;if(returning)returning.textContent="Continue to build";}}catch(error){announce(error.message);}finally{submit.disabled=false;}
  });
  list.addEventListener("click",async(event)=>{
    const pack=event.target.closest(".metadata-pack");if(!pack)return;
    const grouping=event.target.closest("[data-grouping]");
    if(grouping){grouping.disabled=true;try{await post("/dataset/api/metadata/grouping",{...base(pack),action:grouping.dataset.grouping});announce("Pack boundaries saved. Reloading the prefilled wizard.");window.location.reload();}catch(error){announce(error.message);grouping.disabled=false;}return;}
    const exporter=event.target.closest(".export-metadata");
    if(exporter){if(!window.confirm("Export source.yaml and LICENSE.txt into this source pack without overwriting existing files?"))return;exporter.disabled=true;try{const result=await post("/dataset/api/metadata/export",base(pack));announce(`Export complete. Written: ${result.written.join(", ")||"none"}. Existing files preserved: ${result.skipped_existing.join(", ")||"none"}.`);}catch(error){announce(error.message);}finally{exporter.disabled=false;}}
  });
  if(complete)complete.addEventListener("click",async()=>{complete.disabled=true;if(cancel)cancel.disabled=true;try{const result=await post("/dataset/api/metadata/complete",{approval_id:approvalId});announce(result.message);}catch(error){announce(error.message);complete.disabled=false;if(cancel)cancel.disabled=false;}});
  if(cancel)cancel.addEventListener("click",async()=>{cancel.disabled=true;if(complete)complete.disabled=true;try{const result=await post("/dataset/api/metadata/cancel",{approval_id:approvalId});announce(result.message);}catch(error){announce(error.message);cancel.disabled=false;}});
})();
