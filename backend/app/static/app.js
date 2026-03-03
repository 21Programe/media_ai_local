function showTab(name){
  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.getElementById('tab-' + name).classList.remove('hidden');
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

showTab('img');

async function genImage(){
  const payload = {
    prompt: document.getElementById('prompt').value,
    negative_prompt: document.getElementById('neg').value,
    width: parseInt(document.getElementById('w').value || "768",10),
    height: parseInt(document.getElementById('h').value || "768",10),
    steps: parseInt(document.getElementById('steps').value || "30",10),
    cfg: parseFloat(document.getElementById('cfg').value || "7"),
    seed: parseInt(document.getElementById('seed').value || "0",10),
  };
  const r = await fetch('/generate/image', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  document.getElementById('out-img').textContent = await r.text();
}

async function genImg2Img(){
  const f = document.getElementById('file2').files[0];
  if(!f){ alert('Escolha uma imagem'); return; }

  const fd = new FormData();
  fd.append('prompt', document.getElementById('prompt2').value);
  fd.append('negative_prompt', document.getElementById('neg2').value);
  fd.append('strength', document.getElementById('strength2').value);
  fd.append('steps', document.getElementById('steps2').value);
  fd.append('cfg', document.getElementById('cfg2').value);
  fd.append('seed', document.getElementById('seed2').value);
  fd.append('image', f);

  const r = await fetch('/generate/img2img', {method:'POST', body:fd});
  document.getElementById('out-img2img').textContent = await r.text();
}

async function genVideo(){
  const f = document.getElementById('filev').files[0];
  if(!f){ alert('Escolha uma imagem'); return; }

  const fd = new FormData();
  fd.append('image', f);
  fd.append('frames', document.getElementById('frames').value);
  fd.append('fps', document.getElementById('fps').value);
  fd.append('motion_bucket_id', document.getElementById('mb').value);
  fd.append('noise_aug_strength', document.getElementById('noise').value);
  fd.append('seed', document.getElementById('seedv').value);

  const r = await fetch('/generate/video_from_image', {method:'POST', body:fd});
  document.getElementById('out-vid').textContent = await r.text();
}

async function loadHistory(){
  const r = await fetch('/history?limit=50');
  document.getElementById('out-hist').textContent = await r.text();
}

async function doSearch(){
  const q = encodeURIComponent(document.getElementById('q').value);
  const r = await fetch(`/search?q=${q}&k=10`);
  document.getElementById('out-search').textContent = await r.text();
}