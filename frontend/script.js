'use strict';

const $ = (id) => document.getElementById(id);
const state = { view: 'discover', page: 1, limit: 24, total: 0, query: '', minPrice: '', maxPrice: '', customersLoaded: false };

/* ---------- helpers ---------- */
async function api(path, params = {}) {
  const url = new URL('/api' + path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => { if (v !== '' && v !== null && v !== undefined) url.searchParams.set(k, v); });
  let res;
  try { res = await fetch(url); } catch (e) { throw new Error('Cannot reach the server. Is it running?'); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === 'string' ? detail : 'Request failed (' + res.status + ')');
  }
  return data;
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([k, v]) => {
    if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node.setAttribute(k, v);
  });
  children.flat().forEach((c) => { if (c) node.append(c); });
  return node;
}

function thumb(product) {
  const box = el('div', { class: 'thumb' });
  const fallback = () => { box.replaceChildren(el('span', { text: 'No image available' })); };
  if (!product.image_url) { fallback(); return box; }
  const img = el('img', { src: product.image_url, alt: product.title, loading: 'lazy' });
  img.addEventListener('error', fallback);
  box.append(img);
  return box;
}

function priceNode(p) {
  return p.price === null || p.price === undefined
    ? el('p', { class: 'price none', text: 'Price not available' })
    : el('p', { class: 'price', text: '$' + p.price.toFixed(2) });
}

function bar(label, value) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const fill = el('span', { class: 'fill' });
  fill.style.width = pct + '%';
  return el('div', { class: 'bar', 'aria-label': label + ' ' + pct + ' percent' }, el('span', { text: label }), el('span', { class: 'track' }, fill));
}

function card(p) {
  const body = el('div', { class: 'card-body' },
    el('h3', { class: 'card-title', text: p.title }),
    p.brand ? el('p', { class: 'brand-line', text: p.brand }) : null,
    priceNode(p));
  if (p.source && p.source !== 'hybrid') body.append(el('span', { class: 'chip', text: p.source.replace(/_/g, ' ') }));
  if (p.source === 'hybrid') {
    body.append(el('div', { class: 'bars' }, bar('Customers', p.cf_score || 0), bar('Text match', p.content_score || 0)));
  }
  if (p.reason) body.append(el('p', { class: 'why', text: p.reason }));
  const more = el('button', { class: 'more', type: 'button', text: 'See similar items' });
  more.addEventListener('click', () => openDetail(p.item_id));
  body.append(more);
  return el('article', { class: 'card' }, thumb(p), body);
}

function setStatus(id, text, isError) {
  const node = $(id);
  node.textContent = text || '';
  node.classList.toggle('error', !!isError);
}

function showSkeleton(grid, n = 8) {
  grid.replaceChildren(...Array.from({ length: n }, () => el('div', { class: 'skeleton' })));
}

function fillGrid(grid, items, emptyText) {
  if (!items.length) { grid.replaceChildren(el('div', { class: 'empty', text: emptyText })); return; }
  grid.replaceChildren(...items.map(card));
}

/* ---------- views ---------- */
function updateSearchUI() {
  const active = !!(state.query || state.minPrice || state.maxPrice);
  $('clear-search').hidden = !active;
  $('hero').hidden = state.view !== 'discover' || !!state.query;   // give the results the screen while searching
}

function switchView(name) {
  state.view = name;
  document.querySelectorAll('.view').forEach((v) => { v.hidden = v.id !== 'view-' + name; });
  document.querySelectorAll('.tab').forEach((t) => {
    if (t.dataset.view === name) t.setAttribute('aria-current', 'page'); else t.removeAttribute('aria-current');
  });
  updateSearchUI();
  if (name === 'foryou' && !state.customersLoaded) loadCustomers();
  if (name === 'results') loadResults();
}

async function loadDiscover() {
  updateSearchUI();
  const grid = $('discover-grid');
  showSkeleton(grid);
  setStatus('discover-status', 'Loading products...');
  try {
    const data = await api('/products', { q: state.query, min_price: state.minPrice, max_price: state.maxPrice, page: state.page, limit: state.limit });
    state.total = data.total;
    fillGrid(grid, data.items, 'No products match your search. Try fewer words or a wider budget.');
    let msg = (state.query ? 'Results for "' + state.query + '": ' : '') + data.total.toLocaleString() + ' products';
    if (data.excluded_missing_price) msg += ' (' + data.excluded_missing_price.toLocaleString() + ' matching items hidden because they have no price)';
    setStatus('discover-status', msg);
    const pages = Math.max(1, Math.ceil(data.total / state.limit));
    $('page-label').textContent = 'Page ' + state.page + ' of ' + pages;
    $('prev').disabled = state.page <= 1;
    $('next').disabled = state.page >= pages;
  } catch (e) {
    grid.replaceChildren();
    setStatus('discover-status', e.message, true);
  }
}

async function loadCustomers() {
  const select = $('customer');
  try {
    const data = await api('/customers');
    select.replaceChildren(...data.customers.map((c) => {
      const label = c.customer_id.slice(0, 8) + '... (' + c.n_interactions + ' reviews)';
      return el('option', { value: c.customer_id, text: label });
    }));
    if (!data.customers.length) select.replaceChildren(el('option', { value: '', text: 'No eligible customers' }));
    state.customersLoaded = true;
    state.customers = data.customers;
  } catch (e) {
    select.replaceChildren(el('option', { value: '', text: 'Unavailable' }));
    setStatus('foryou-status', e.message, true);
  }
}

async function loadRecs() {
  const id = $('customer').value;
  const grid = $('foryou-grid');
  if (!id) { setStatus('foryou-status', 'Choose a customer first.', true); return; }
  const c = (state.customers || []).find((x) => x.customer_id === id);
  $('liked').textContent = c && c.liked_examples.length ? 'This customer liked: ' + c.liked_examples.join(' | ') : '';
  showSkeleton(grid, 6);
  setStatus('foryou-status', 'Finding recommendations...');
  try {
    const data = await api('/hybrid/' + encodeURIComponent(id), { weight: $('weight').value, max_price: $('fy-max-price').value, k: 12 });
    fillGrid(grid, data.items, data.message || 'No recommendations available.');
    let msg = data.message || '';
    if (data.excluded_missing_price) msg += ' ' + data.excluded_missing_price + ' suggestions hidden because they have no price.';
    setStatus('foryou-status', msg.trim());
  } catch (e) {
    grid.replaceChildren();
    setStatus('foryou-status', e.message, true);
  }
}

async function loadResults() {
  const body = $('results-body');
  setStatus('results-status', 'Loading results...');
  try {
    const m = await api('/metrics');
    const k = m.k;
    const cols = ['precision@' + k, 'recall@' + k, 'map@' + k, 'ndcg@' + k, 'coverage', 'n_users'];
    const names = { popularity: 'Popularity baseline', content: 'Content-based', collaborative: 'Collaborative (SVD)', hybrid: 'Hybrid' };
    const best = Math.max(...Object.values(m.test).map((r) => r['ndcg@' + k]));
    const table = el('table', {},
      el('thead', {}, el('tr', {}, el('th', { text: 'Model' }), cols.map((c) => el('th', { text: c.replace('n_users', 'users') })))),
      el('tbody', {}, Object.keys(names).filter((n) => m.test[n]).map((n) => {
        const r = m.test[n];
        const row = el('tr', r['ndcg@' + k] === best ? { class: 'best' } : {}, el('td', { text: names[n] }),
          cols.map((c) => el('td', { text: c === 'n_users' ? String(r[c]) : r[c].toFixed(4) })));
        return row;
      })));
    body.replaceChildren(
      el('div', { class: 'table-wrap' }, table),
      el('p', { class: 'hint', text: 'Settings chosen on validation data: ' + m.chosen.svd_components + ' SVD factors, collaborative weight ' + m.chosen.hybrid_cf_weight + '. Test users scored: ' + m.eligible_users.test + '. Highlighted row has the best NDCG@' + k + '.' }),
      el('p', { class: 'hint', text: 'Positive = rating of ' + m.positive_rating + ' or more. A model that loses to the popularity baseline is reported as it is.' }));
    setStatus('results-status', '');
  } catch (e) {
    body.replaceChildren();
    setStatus('results-status', e.message, true);
  }
}

/* ---------- product dialog ---------- */
async function openDetail(id) {
  const dlg = $('detail');
  const body = $('detail-body');
  body.replaceChildren(el('p', { text: 'Loading...' }));
  if (!dlg.open) dlg.showModal();
  try {
    const data = await api('/products/' + encodeURIComponent(id) + '/similar', { k: 6 });
    const p = data.product;
    const top = el('div', { class: 'detail-top' }, thumb(p),
      el('div', {}, el('h3', { id: 'detail-title', text: p.title }),
        p.brand ? el('p', { text: 'Brand: ' + p.brand }) : el('p', { text: 'Brand not listed' }),
        p.category ? el('p', { text: 'Category: ' + p.category }) : null,
        priceNode(p),
        p.features ? el('p', { class: 'desc', text: p.features }) : null,
        p.description ? el('p', { class: 'desc', text: p.description }) : el('p', { text: 'No description available for this product.' })));
    const sim = el('div', {}, el('h3', { text: 'Similar products' }));
    if (data.items.length) sim.append(el('div', { class: 'grid' }, data.items.map(card)));
    else sim.append(el('p', { class: 'hint', text: data.message || 'No similar products found.' }));
    if (data.variants_hidden) sim.append(el('p', { class: 'hint', text: data.variants_hidden + ' near-identical listings (other colours or sizes of the same product) are hidden.' }));
    body.replaceChildren(top, sim);
  } catch (e) {
    body.replaceChildren(el('p', { class: 'status error', text: e.message }));
  }
}

/* ---------- wiring ---------- */
$('search-form').addEventListener('submit', (e) => {
  e.preventDefault();
  state.query = $('q').value.trim(); state.page = 1;
  switchView('discover');
  loadDiscover().then(() => $('discover-status').scrollIntoView({ behavior: 'smooth', block: 'start' }));
});
$('clear-search').addEventListener('click', () => {
  state.query = ''; state.minPrice = ''; state.maxPrice = ''; state.page = 1;
  $('q').value = ''; $('min-price').value = ''; $('max-price').value = '';
  loadDiscover();
});
$('apply-filters').addEventListener('click', () => {
  state.minPrice = $('min-price').value; state.maxPrice = $('max-price').value; state.page = 1; loadDiscover();
});
$('prev').addEventListener('click', () => { state.page -= 1; loadDiscover(); });
$('next').addEventListener('click', () => { state.page += 1; loadDiscover(); });
$('weight').addEventListener('input', () => { $('weight-out').textContent = $('weight').value; });
$('load-recs').addEventListener('click', loadRecs);
$('detail-close').addEventListener('click', () => $('detail').close());
$('detail').addEventListener('click', (e) => { if (e.target === $('detail')) $('detail').close(); });
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchView(t.dataset.view)));

switchView('discover');
loadDiscover();