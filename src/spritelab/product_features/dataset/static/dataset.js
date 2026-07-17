(() => {
  "use strict";
  const choose = document.getElementById("choose-folder");
  const build = document.getElementById("build-dataset");
  const chooseExisting = document.getElementById("choose-existing-dataset");
  const result = document.getElementById("dataset-summary");
  const selected = document.getElementById("selected-folder");
  const disabledReason = document.getElementById("build-disabled-reason");
  const actions = document.getElementById("dataset-actions");
  const reviewAction = document.getElementById("dataset-review-action");
  const metadataAction = document.getElementById("dataset-metadata-action");
  if (!choose || !build || !result || !selected) return;
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  let approvalId = new URLSearchParams(window.location.search).get("approval_id");
  let existingDataset = null;
  const request = async (url, payload) => {
    const options = {
      method:"POST",
      headers:{"Content-Type":"application/json","X-CSRF-Token":csrf},
      body:JSON.stringify(payload || {}),
    };
    const response = await fetch(url, payload === undefined ? {} : options);
    const contentType=response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) { await response.text(); throw new Error("Sprite Lab received an unexpected response. Reload and try again."); }
    const body=await response.json();
    if(!response.ok) throw new Error(body.message || "The folder action could not be completed.");
    return body;
  };
  const sleep = (milliseconds) => new Promise(resolve => window.setTimeout(resolve, milliseconds));
  const showJob = (job) => {
    const lines = (job.logs || []).map(entry => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      return `[${time}] ${entry.message}`;
    });
    result.textContent = `${job.message || "Dataset build is running."}\n\n${lines.join("\n")}`;
  };
  const showSelectionJob = (job) => {
    const lines=(job.logs||[]).map(entry=>`[${new Date(entry.timestamp).toLocaleTimeString()}] ${entry.message}`);
    result.textContent=`${job.message||"Dataset selection is running."}\n\n${lines.join("\n")}`;
  };
  const busy=(node,value)=>{
    node.disabled=value;
    if(value)node.setAttribute("aria-busy","true");else node.removeAttribute("aria-busy");
  };
  const rememberApproval=()=>{
    const url=new URL(window.location.href);
    if(approvalId)url.searchParams.set("approval_id",approvalId);else url.searchParams.delete("approval_id");
    window.history.replaceState({},"",url);
  };
  const applyInspection=(inspection)=>{
    existingDataset=null;
    build.textContent="Build dataset";
    if(actions)actions.hidden=true;
    selected.textContent=inspection.folder_name||"Selected folder";
    build.disabled=!inspection.image_count||inspection.wizard_required;
    if(metadataAction){metadataAction.hidden=!inspection.wizard_required;metadataAction.href=`/dataset/metadata?approval_id=${encodeURIComponent(approvalId)}`;}
    disabledReason.textContent=!inspection.image_count ? "The selected folder contains no PNG images." : inspection.wizard_required ? "Complete one declaration for each incomplete pack first." : "Folder is ready for an explicit build.";
    result.textContent=`${inspection.image_count} PNG file(s) in ${inspection.pack_count} pack(s). ${inspection.wizard_required ? "Some packs need source or license information." : "Pack evidence is ready."}\nNext action: ${inspection.next_action}`;
  };
  const applyExisting=(existing,{restored=false}={})=>{
    approvalId=null;existingDataset=existing;rememberApproval();selected.textContent=existing.folder_name||"Selected dataset";
    build.disabled=false;build.removeAttribute("aria-busy");build.textContent="Use selected dataset";
    disabledReason.textContent="The imported dataset is validated. Click Use selected dataset to continue without rebuilding.";
    result.textContent=`${existing.item_count||0} imported item(s). ${restored?"Restored the active imported dataset.":existing.message}\nNext action: Use selected dataset.`;
    if(actions)actions.hidden=true;
    if(reviewAction)reviewAction.href=existing.review_url||"/dataset/review";
    if(metadataAction)metadataAction.hidden=true;
  };
  choose.addEventListener("click", async () => {
    busy(choose,true); result.textContent="Opening the native folder picker…";
    try {
      const choice=await request("/dataset/api/folders/choose",{});
      approvalId=choice.approval.approval_id; selected.textContent=choice.approval.folder_name;rememberApproval();
      result.textContent="Checking images, source records, and license records…";
      const inspection=await request("/dataset/api/inspect",{approval_id:approvalId});
      applyInspection(inspection);
      if(inspection.wizard_required){window.location.assign(`/dataset/metadata?approval_id=${encodeURIComponent(approvalId)}`);}
    } catch(error) { approvalId=null;rememberApproval();build.disabled=true;disabledReason.textContent="Choose and inspect a folder first.";result.textContent=`${error.message}\nNext action: Choose image folder`; }
    finally { busy(choose,false); }
  });
  build.addEventListener("click", async () => {
    if(existingDataset){
      busy(build,true);
      try {
        result.textContent=`${existingDataset.item_count||0} imported item(s). The selected dataset is active and no rebuild was performed.\nOpening dataset reviewâ€¦`;
        if(actions)actions.hidden=false;
        if(reviewAction)reviewAction.href=existingDataset.review_url||"/dataset/review";
        window.location.assign(existingDataset.review_url||"/dataset/review");
      } finally { busy(build,false); }
      return;
    }
    if(!approvalId)return;busy(build,true);result.textContent="Starting dataset build in the background…\nYour original files will not be changed.";
    try {
      const response=await request("/dataset/api/build",{approval_id:approvalId,confirm_hosted:document.getElementById("confirm-hosted")?.checked===true});
      let job;
      do {
        await sleep(500);
        job=await request(response.status_url);
        showJob(job);
      } while(job.status==="queued" || job.status==="running");
      if(job.status==="failed")throw new Error(job.message || "Dataset build failed.");
      const data=job.result?.data||{};
      const counts=data.counts||{};
      result.textContent+=`\n\n${counts.processed||0} images processed\n${counts.accepted||0} accepted\n${counts.rejected_automatically||counts.rejected||0} rejected\n${counts.uncertain||0} uncertain\nNext action: ${(counts.excluded||counts.rejected)?"Rescue images":"Review dataset status"}`;
      if(actions)actions.hidden=false;
      if(reviewAction&&data.review_url)reviewAction.href=data.review_url;
    }
    catch(error){result.textContent+=`\n\n${error.message}\nNext action: Try the build again.`;}
    finally{busy(build,false);}
  });
  chooseExisting?.addEventListener("click", async () => {
    busy(chooseExisting,true);result.textContent="Starting dataset selection\u2026\nThe application remains available while the folder is checked.";
    try {
      const response=await request("/dataset/api/existing/choose",{});
      let job;
      do {
        await sleep(350);
        job=await request(response.status_url);
        showSelectionJob(job);
      } while(job.status==="queued" || job.status==="running");
      if(job.status==="failed")throw new Error(job.message||"The existing dataset could not be selected.");
      const existing=job.result||{};
      const logs=result.textContent;
      applyExisting(existing);
      result.textContent=`${logs}\n\n${result.textContent}`;
    } catch(error) {
      existingDataset=null;build.disabled=true;build.removeAttribute("aria-busy");build.textContent="Build dataset";
      result.textContent=`${error.message}\nNext action: Select a valid imported dataset folder.`;
    }
    finally { busy(chooseExisting,false); }
  });
  if(approvalId){
    result.textContent="Restoring the selected folder…";
    request("/dataset/api/inspect",{approval_id:approvalId}).then(applyInspection).catch(error=>{
      approvalId=null;rememberApproval();build.disabled=true;disabledReason.textContent="Choose and inspect a folder first.";result.textContent=`${error.message}\nNext action: Choose image folder`;
    });
  }else{
    request("/dataset/api/selection").then(selection=>{
      if(selection.mode==="source"){
        approvalId=selection.approval_id;rememberApproval();applyInspection(selection);
        result.textContent+=`\nRestored the previously selected source folder.`;
      }else if(selection.mode==="existing")applyExisting(selection,{restored:true});
    }).catch(()=>{});
  }
})();
