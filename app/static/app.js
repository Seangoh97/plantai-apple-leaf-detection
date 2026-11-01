'use strict';
const $ = id => document.getElementById(id);
let selected = null, objectURL = null, ready = false, busy = false;
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function updateButton() { $('analyze').disabled = !selected || !ready || busy; }
function clearResult() { $('results').hidden = true; $('empty').hidden = false; }
async function checkModel() {
  try {
    const response = await fetch('/api/health', {cache: 'no-store'});
    if (!response.ok) throw new Error();
    const data = await response.json();
    ready = data.ready;
    $('model-status').textContent = ready ? '● Model connected · ' + data.input_size.join(' × ') + ' RGB · ' + (data.threshold_source === 'saved' ? 'Saved thresholds' : 'Default 0.50 thresholds') : '○ Setup needed · ' + data.message;
    $('model-status').classList.toggle('ready', ready);
  } catch { ready = false; $('model-status').textContent = '○ Server unavailable. Check that FastAPI is running.'; }
  updateButton();
}
function clearPhoto() {
  selected = null;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  $('preview').removeAttribute('src'); $('preview').hidden = true;
  $('file-name').hidden = true; $('upload-prompt').hidden = false;
  $('reset').hidden = true; $('file').value = ''; $('capture').value = '';
  clearResult(); showError(''); updateButton();
}
function selectPhoto(file) {
  if (busy) return;
  clearPhoto();
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { showError('Choose a JPEG, PNG or WebP photo. Convert HEIC to JPEG first.'); return; }
  if (file.size > 10 * 1024 * 1024 || file.size === 0) { showError('Choose a nonempty photo of 10 MB or smaller.'); return; }
  objectURL = URL.createObjectURL(file);
  const currentURL = objectURL;
  $('preview').onload = () => {
    if (objectURL !== currentURL) return;
    const w = $('preview').naturalWidth, h = $('preview').naturalHeight;
    if (w * h > 20000000 || Math.min(w,h) < 32) { clearPhoto(); showError('Use an image between 32 × 32 pixels and 20 megapixels.'); return; }
    selected = file; $('preview').hidden = false; $('upload-prompt').hidden = true;
    $('file-name').textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(1) + ' MB';
    $('file-name').hidden = false; $('reset').hidden = false; updateButton();
  };
  $('preview').onerror = () => { if (objectURL === currentURL) { clearPhoto(); showError('Your browser could not read this photo. Try exporting it as JPEG.'); } };
  $('preview').src = objectURL;
}
$('choose').onclick = () => $('file').click();
$('camera').onclick = () => $('capture').click();
$('file').onchange = event => selectPhoto(event.target.files[0]);
$('capture').onchange = event => selectPhoto(event.target.files[0]);
$('reset').onclick = clearPhoto;
const zone = $('dropzone');
for (const type of ['dragenter', 'dragover']) zone.addEventListener(type, event => { event.preventDefault(); if (!busy) zone.classList.add('dragover'); });
for (const type of ['dragleave', 'drop']) zone.addEventListener(type, event => { event.preventDefault(); zone.classList.remove('dragover'); });
zone.addEventListener('drop', event => { if (event.dataTransfer.files.length !== 1) { showError('Choose one photo at a time.'); return; } selectPhoto(event.dataTransfer.files[0]); });
function renderResult(data) {
  $('summary').textContent = data.summary;
  $('result-kicker').textContent = data.status === 'review_needed' ? 'Review recommended' : 'Model signals · verify with a specialist';
  $('probabilities').replaceChildren();
  for (const row of data.classes) {
    const wrapper = document.createElement('div'); wrapper.className = 'prob-row';
    const heading = document.createElement('div'); heading.className = 'prob-heading';
    const name = document.createElement('span'); name.textContent = row.label;
    const value = document.createElement('span'); value.className = 'prob-value'; value.textContent = (row.probability * 100).toFixed(1) + '%';
    heading.append(name, value);
    const progress = document.createElement('progress'); progress.className = 'track'; progress.max = 1; progress.value = row.probability; progress.setAttribute('aria-label', row.label + ' probability');
    const meta = document.createElement('div'); meta.className = 'prob-meta';
    const threshold = document.createElement('span'); threshold.textContent = 'Threshold ' + (row.threshold * 100).toFixed(1) + '%';
    const signal = document.createElement('span'); signal.textContent = row.detected ? 'Threshold met' : 'Below threshold'; if (row.detected) signal.className = 'detected';
    meta.append(threshold, signal); wrapper.append(heading, progress, meta); $('probabilities').append(wrapper);
  }
  $('reasons').replaceChildren();
  const reasons = data.uncertainty.reasons.length ? data.uncertainty.reasons : ['No heuristic review flag was triggered. This does not guarantee that the result is correct.'];
  for (const reason of reasons) { const item = document.createElement('li'); item.textContent = reason; $('reasons').append(item); }
  $('threshold-note').textContent = data.threshold_source === 'saved' ? 'Using your saved per-class decision thresholds.' : 'Using default 0.50 thresholds, which have not been tuned for this deployment.';
  $('empty').hidden = true; $('results').hidden = false;
}
$('analyze').onclick = async () => {
  if (!selected || !ready || busy) return;
  busy = true; showError(''); clearResult(); updateButton();
  for (const id of ['choose','camera','reset']) $(id).disabled = true;
  $('analyze').textContent = 'Analyzing your leaf…'; $('results-title').setAttribute('aria-busy','true');
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 90000);
  try {
    const form = new FormData(); form.append('file', selected);
    const response = await fetch('/api/predict', {method:'POST', body:form, signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The server rejected this upload. Choose a valid photo.');
    renderResult(data);
  } catch (error) { showError(error.name === 'AbortError' ? 'The request timed out. Your server may still be processing it. Wait a moment before retrying.' : (error.message || 'Unable to connect to the server.')); }
  finally {
    clearTimeout(timer); busy = false; $('analyze').textContent = 'Analyze leaf →';
    for (const id of ['choose','camera','reset']) $(id).disabled = false;
    $('results-title').removeAttribute('aria-busy'); await checkModel(); updateButton();
  }
};
checkModel();
setInterval(() => { if (!busy) checkModel(); }, 15000);
window.addEventListener('beforeunload', () => { if (objectURL) URL.revokeObjectURL(objectURL); });
