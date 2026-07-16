(() => {
  "use strict";
  const choose = document.getElementById("choose-folder");
  const build = document.getElementById("build-dataset");
  const result = document.getElementById("dataset-summary");
  const selected = document.getElementById("selected-folder");
  const disabledReason = document.getElementById("build-disabled-reason");
  if (!choose || !build || !result || !selected) return;
  const csrf = document.querySelector('meta[name="spritelab-csrf"]')?.content || "";
  let approvalId = null;
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
  const busy=(node,value)=>{node.disabled=value;node.setAttribute("aria-busy",String(value));};
  choose.addEventListener("click", async () => {
    busy(choose,true); result.textContent="Opening the native folder picker…";
    try {
      const choice=await request("/dataset/api/folders/choose",{});
      approvalId=choice.approval.approval_id; selected.textContent=choice.approval.folder_name;
      result.textContent="Checking images, source records, and license records…";
      const inspection=await request("/dataset/api/inspect",{approval_id:approvalId});
      build.disabled=!inspection.image_count||inspection.wizard_required; disabledReason.textContent=!inspection.image_count ? "The selected folder contains no PNG images." : inspection.wizard_required ? "Complete one declaration for each incomplete pack first." : "Folder is ready for an explicit build.";
      result.textContent=`${inspection.image_count} PNG file(s) in ${inspection.pack_count} pack(s). ${inspection.wizard_required ? "Some packs need source or license information." : "Pack evidence is ready."}\nNext action: ${inspection.next_action}`;
      if(inspection.wizard_required){window.location.assign(`/dataset/metadata?approval_id=${encodeURIComponent(approvalId)}`);}
    } catch(error) { approvalId=null;build.disabled=true;disabledReason.textContent="Choose and inspect a folder first.";result.textContent=`${error.message}\nNext action: Choose image folder`; }
    finally { busy(choose,false); }
  });
  build.addEventListener("click", async () => {
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
    }
    catch(error){result.textContent+=`\n\n${error.message}\nNext action: Try the build again.`;}
    finally{busy(build,false);}
  });
})();
