'use strict';
const $ = id => document.getElementById(id);
const BODIES = ['Franka', 'UR5e', 'xArm7', 'Kinova', 'Sawyer'];
const TASK_NAMES = new Map([
  ['hang_cup', '挂杯'],
  ['pick_mugcup_place_coffeemachine', '咖啡机放杯'],
  ['pick_leek_place_cuttingboard', '韭菜放切板'],
]);
const TASK_ALIASES = new Map([
  ['coffee', 'pick_mugcup_place_coffeemachine'],
  ['leek', 'pick_leek_place_cuttingboard'],
]);

const DINO_V2 = 'DINOv2-B/14-reg4';
const RESNET = 'ResNet18（ImageNet 初始化）';
const RAW_RESNET = '原始 ' + RESNET;
const NO_ALIGNMENT = '否 · 无换臂预对齐';
const ALIGNMENT = '是 · 换臂特征对齐';
const ALIGNMENT_GEOMETRY = '是 · 换臂特征对齐＋几何辅助';
const GEOMETRY_ONLY = '否 · 仅几何辅助预训练';
const FROZEN = '冻结';
const JOINT = '不冻结 · 随动作策略更新';
const UNKNOWN_VISUAL = ['未核实', '未核实', '未核实'];

// Bind training recipes to record IDs because inference freeze flags do not
// establish how the encoder was trained.
const VISUAL_PROFILES = new Map([
  // LeRobot ablations and aligned-to-joint ACT use different checkpoints.
  [[
    'lerobot24_dino_align_geometry', 'completed24_eval104_coffee_align_geometry',
    'completed24_eval104_unified_oldgoal_aux', 'completed24_eval104_unified_irg_affsup',
    'requested_temporal_hang_align_geometry', 'temporal_dinov2', 'temporal_canonical_D',
    'dp_transfer104_hang_dino_dp',
  ], DINO_V2, ALIGNMENT_GEOMETRY, FROZEN],
  [[
    'lerobot24_dino_align_only', 'requested_temporal_hang_align_only', 'dino_act_flow24_aligned_dino_flow',
  ], DINO_V2, ALIGNMENT, FROZEN],
  [[
    'requested_temporal_hang_align_geometry_self',
  ], DINO_V2, '是 · 换臂特征对齐＋几何辅助＋源臂自特征监督', FROZEN],
  [[
    'dino_act_flow24_public_dino_act', 'dino_act_flow24_public_dino_flow',
  ], DINO_V2 + '（公开原权重）', NO_ALIGNMENT, FROZEN],
  [['lerobot24_dino_geometry_only'], DINO_V2, GEOMETRY_ONLY, FROZEN],
  [['lerobot_unfrozen24_dino_align_only'], DINO_V2, ALIGNMENT, JOINT],
  [['hang_dino_joint_dp24_gpu0'], DINO_V2, ALIGNMENT_GEOMETRY, JOINT],
  [[
    'lerobot24_resnet_align_geometry', 'temporal_resnet',
  ], RESNET, ALIGNMENT_GEOMETRY, FROZEN],
  [['lerobot24_resnet_align_only'], RESNET, ALIGNMENT, FROZEN],
  [['lerobot24_resnet_geometry_only'], RESNET, GEOMETRY_ONLY, FROZEN],
  [['lerobot_unfrozen24_resnet_align_only'], RESNET, ALIGNMENT, JOINT],
  [['lerobot24_resnet_franka400', 'requested_temporal_rgb_act_franka400'], RAW_RESNET, NO_ALIGNMENT, JOINT],
  [[
    'dp_transfer104_hang_resnet_dp', 'dp_transfer104_coffee_plain_dp', 'history_single_raw400_aghost_dp',
    'aghost_franka400_hang_plain',
  ], '原始 ResNet18（从零训练）', NO_ALIGNMENT, JOINT],
  [['aghost_franka100_coffee_frozen', 'aghost_franka400_hang_frozen'],
    '双视角 ResNet18', ALIGNMENT, FROZEN],
  [['aghost_franka100_coffee_joint', 'aghost_franka400_hang_joint'],
    '双视角 ResNet18', ALIGNMENT, JOINT],
  // The 133 temporal comparison uses the distilled encoder.
  [['rgbact133_temporal_dinov2'], DINO_V2, '是 · 换臂特征蒸馏', FROZEN],
  [['rgbact133_temporal_resnet'], RESNET, '是 · 换臂特征蒸馏', FROZEN],
  [[
    'pick_dino24_coffee_aligned', 'pick_dino24_leek_aligned',
  ], DINO_V2, ALIGNMENT, FROZEN],
  [[
    'pick_dino24_coffee_baseline', 'pick_dino24_leek_baseline',
  ], DINO_V2 + '（公开原权重初始化）', NO_ALIGNMENT, JOINT],
  [['pick_dino24_leek_pair_preparation'], '不适用 · 配对 RGB 渲染', '不适用', '不适用'],
  // Historical lanes retain the distinction between planner and action encoders.
  [[
    'requested_temporal_irg_affsup', 'requested_temporal_irg_oldgoal_aux',
    'history_multi_raw400_finetune', 'history_multi_raw400_end_to_end',
    'history_multi_skeleton400_finetune', 'history_single_raw400_finetune',
  ], 'PaliGemma 图像塔（规划）＋原始 ResNet18（ACT）', NO_ALIGNMENT, 'PaliGemma 冻结；ACT 的 ResNet18 不冻结'],
  [[
    'history_single_raw400_act', 'history_single_raw400_dp',
    'history_multi_raw400_act', 'history_multi_raw400_dp',
    'history_multi_skeleton400_act', 'history_multi_skeleton400_dp',
  ], RAW_RESNET, NO_ALIGNMENT, JOINT],
  [[
    'history_multi_raw400_unified_fm', 'history_multi_raw400_unified_fm50000',
    'history_multi_raw400_unified_act50000', 'history_single_raw400_pretrained_fm',
    'history_single_raw400_pretrained_act',
  ], 'DINOv3 H+/16＋视觉适配器', '是 · 同帧跨本体特征蒸馏', 'DINOv3 骨干冻结；视觉适配器不冻结'],
  [[
    'history_single_raw400_aghost_se3', 'history_single_raw400_aghost_affordance_goal',
    'history_single_raw400_aghost_affordance_goal_dp',
  ], '原始 DINOv3 H+/16（高层）＋ResNet18（动作）', NO_ALIGNMENT,
  '高层 DINOv3 冻结；动作 ResNet18 从零训练、不冻结'],
].flatMap(([ids, ...profile]) => ids.map(id => [id, profile])));

function visualProfile(id) {
  if (/^affordance_goal24_public_[AB]_\d+$/.test(id)) {
    return [DINO_V2 + '（公开原权重）', NO_ALIGNMENT, FROZEN];
  }
  if (/^affordance_goal24_(?:(?:dp_)?[AB]|(?:dp)?v2)_\d+$/.test(id)) {
    return [DINO_V2, ALIGNMENT, FROZEN];
  }
  return VISUAL_PROFILES.get(id) ?? UNKNOWN_VISUAL;
}
const STATES = {testing:'测试中',training:'训练中',running:'运行中',complete:'已完成',stopped:'已停止',
  error:'异常',waiting:'等待',unknown:'待核实',confirming:'等待收尾',stale:'数据过期',paused:'已暂停',
  partial:'部分完成 / 分片',trained:'训练完成',queued:'排队中',failed:'失败'};
const escapeHTML = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));
let snapshot = null, selectedMetric = 'both', fetching = false, refreshError = null;
const dashboardConfig = typeof document === 'undefined' ? {} : document.documentElement.dataset;
const refreshSeconds = Number(dashboardConfig.refreshSeconds) || 15;
let lastApiRefresh = 0;
const REMOVED_RECORDS_KEY = 'afta.visual-dashboard.removed-records';
let editing = false, removedRecords = new Set();
const num = x => Number.isFinite(x) ? x.toLocaleString('zh-CN') : '—';
const time = x => x ? new Date(x * 1000).toLocaleString('zh-CN', {hour12:false}) : '未登记';
const beijingTime = new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai',
  month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hourCycle:'h23'});

function methodName(row) {
  if (row.id.startsWith('affordance_goal24_dpv2_')) return 'GoalDiffusionPolicyV2 · DP';
  if (row.id.startsWith('affordance_goal24_dp_')) return 'Affordance + SE(3) Goal · DP';
  if (/^affordance_goal24_(?:public_)?[AB]_/.test(row.id)) return 'Affordance + SE(3) Goal · ACT';
  if (row.id.endsWith('_pair_preparation')) return '数据准备';
  if (/GoalPolicyV2/i.test(row.title)) return 'GoalPolicyV2';
  if (/IRG-AffSup/i.test(row.title)) return 'IRG-AffSup';
  if (/旧式\s*goal/i.test(row.title)) return '旧式 Goal＋辅助监督';
  if (/我们的方法-微调/.test(row.title)) return '我们的方法 · 微调';
  if (/我们的方法-端到端/.test(row.title)) return '我们的方法 · 端到端';
  const labels = [row.title, ...(row.training || []).map(stage => stage.label)].join(' ');
  if (/\b(?:Flow Matching|DINOv3 FM)\b/i.test(labels)) return 'Flow Matching';
  if (/\b(?:Diffusion Policy|DP)\b/i.test(labels)) return 'Diffusion Policy';
  if (/\bACT\b/i.test(labels)) return 'ACT';
  return '方法未注明';
}

function classifyRow(row) {
  const taskKey = TASK_ALIASES.get(row.task) || row.task || 'unknown';
  return {...row, taskKey, taskName: TASK_NAMES.get(taskKey) || row.task_label || '任务未注明',
    methodName: methodName(row), visual: visualProfile(row.id)};
}

function readRemovedRecords() {
  try {
    const ids = JSON.parse(localStorage.getItem(REMOVED_RECORDS_KEY));
    return new Set(Array.isArray(ids) ? ids.filter(id => typeof id === 'string') : []);
  } catch {
    return new Set();
  }
}

function setRecordRemoved(id, remove) {
  const row = snapshot.rows.find(item => item.id === id);
  if (remove && !row) return;
  const ids = readRemovedRecords();
  if (remove) ids.add(id);
  else ids.delete(id);
  try {
    localStorage.setItem(REMOVED_RECORDS_KEY, JSON.stringify([...ids]));
  } catch {
    $('edit-status').textContent = '浏览器无法保存修改，请检查网站数据设置。';
    return;
  }
  removedRecords = ids;
  render();
  $('edit-status').textContent = `已${remove ? '删除' : '恢复'}「${row?.title ?? id}」。`;
}

function renderRecordEditor() {
  const edit = $('edit-records');
  edit.disabled = false;
  edit.textContent = editing ? '完成编辑' : '编辑';
  edit.setAttribute('aria-expanded', String(editing));
  $('record-editor').hidden = !editing;
  $('removed-records').hidden = removedRecords.size === 0;
  if (!editing) return;
  const records = new Map(snapshot.rows.map(row => [row.id, row]));
  $('removed-count').textContent = `已删除的记录（${removedRecords.size}）`;
  $('removed-list').innerHTML = [...removedRecords].map(id => {
    const row = records.get(id), title = escapeHTML(row?.title ?? id);
    return `<li><div>${title}${row ? `<small>${escapeHTML(row.family)}${row.host ? ' · ' + escapeHTML(row.host) : ''}</small>` : ''}
      </div><button data-record-action="restore" data-id="${escapeHTML(id)}" aria-label="恢复 ${title}">恢复</button></li>`;
  }).join('');
}

function percentage(success, count) {
  if (!Number.isFinite(success) || !Number.isFinite(count) || count <= 0 || success < 0 || success > count) return null;
  return Math.round(1000 * success / count) / 10;
}

function metricTotal(rows, metric) {
  let successes = 0, completed = 0, unknown = 0;
  for (const row of rows) {
    if (row.accounting?.conflicts?.length || metric === 'hang' && row.task !== 'hang_cup' ||
        metric === 'place' && row.task === 'hang_cup') continue;
    for (const g of row.groups || []) {
      if (!Number.isFinite(g[metric]) || !Number.isFinite(g.completed) ||
          g[metric] < 0 || g[metric] > g.completed) continue;
      successes += g[metric]; completed += g.completed; unknown += g[metric + '_unknown'] || 0;
    }
  }
  return {successes, completed, unknown, percent:percentage(successes,completed)};
}

function totalGroup(row) {
  const pending = (row.groups || []).reduce((sum, group) => sum + (group.pending || 0), 0);
  const total = {name:'总计', expected:row.expected, completed:0, pending};
  if (row.groups?.some(group => Number.isInteger(group.accepted))) {
    total.accepted = row.groups.reduce((sum, group) => sum + group.accepted, 0);
  }
  for (const metric of ['grasp','hang','place']) {
    const value = metricTotal([row],metric);
    total[metric] = value.completed ? value.successes : null;
    total[metric + '_completed'] = value.completed;
    total[metric + '_unknown'] = value.unknown;
    total.completed = Math.max(total.completed,value.completed);
  }
  if (row.thread_metric) {
    for (const key of ['thread','thread_completed','thread_unknown']) {
      total[key] = (row.groups || []).reduce((sum, group) => sum + (group[key] || 0), 0);
    }
  }
  return total;
}

function renderTotals(rows) {
  for (const metric of ['grasp','hang','place']) {
    const value = metricTotal(rows,metric);
    $('total-' + metric + '-rate').textContent = value.percent === null ? '—' : value.percent + '%';
    $('total-' + metric + '-count').textContent = value.completed ?
      `${num(value.successes)} / ${num(value.completed)} 次${value.unknown ? ` · 未知 ${value.unknown} 次` : ''}` :
      '尚未测试 / 未记录';
  }
}

function measurement(group, metric, task) {
  const label = {grasp:'抓取', thread:'抓取后穿柄', hang:'挂杯', place:'放置'}[metric];
  if ((metric === 'place' && task === 'hang_cup') || (metric === 'hang' && task !== 'hang_cup'))
    return '<div class="measure"><span class="na">不适用</span></div>';
  const count = group?.[metric + '_completed'] ?? group?.completed ?? 0;
  const success = group?.[metric], unknown = group?.[metric + '_unknown'] || 0;
  const rate = percentage(success, count);
  if (rate === null) return `<div class="measure"><div class="na">—</div><div class="fraction">${label}${group?.pending ? '等待入账' : '尚未测试 / 未记录'}</div></div>`;
  return `<div class="measure"><div class="rate"><span class="measure-label">${label}</span>${rate}%</div>
    <div class="fraction">${num(success)} / ${num(count)}${group.expected && metric !== 'thread' ? ` · 计划${num(group.expected)}` : ''}</div>
    ${unknown ? `<div class="unknown">未知 ${unknown} 次</div>` : ''}</div>`;
}

function cell(row, body) {
  if (row.accounting?.conflicts?.length) return '<td class="metric-cell"><span class="unknown">统计待核实</span></td>';
  const group = body === '总计' ? totalGroup(row) : (row.groups || []).find(g => g.name === body);
  const metrics = selectedMetric === 'both' ? ['grasp', row.task === 'hang_cup' ? 'hang' : 'place'] : [selectedMetric];
  if (row.thread_metric && selectedMetric === 'both') metrics.splice(1, 0, 'thread');
  const pending = group?.pending > 0 ? `<div class="fraction">${Number.isInteger(group.accepted) ?
    '其中待正式入账' : '已结束，待入账'} ${num(group.pending)} 场</div>` : '';
  return `<td class="metric-cell">${metrics.map(m => measurement(group, m, row.task)).join('')}${pending}</td>`;
}

function completionLine(label, prediction, fallback, reason) {
  const finish = prediction?.finish_at;
  if (refreshError || snapshotExpired() || !Number.isFinite(finish) || finish <= snapshot.generated_at) {
    const status = refreshError ? '数据刷新失败' : snapshotExpired() ? '数据待刷新' : fallback;
    return `<div><span>${escapeHTML(label)}</span> ${escapeHTML(status)}</div>`;
  }
  const date = new Date(finish * 1000);
  const minutes = Math.ceil((finish - snapshot.generated_at) / 60);
  const remaining = minutes < 60 ? `${minutes} 分钟` :
    `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分钟`;
  return `<div title="${escapeHTML(reason)}"><span>${escapeHTML(label)}</span>
    <time datetime="${date.toISOString()}">${beijingTime.format(date)}</time>
    <small>约剩 ${remaining}</small></div>`;
}

function renderCompletion(row) {
  const estimates = snapshot.completion_estimates?.[row.id] || {};
  const stages = (row.training || []).filter(stage => stage.state !== 'complete');
  const current = stages[0]?.state === 'running' ? stages[0] : null;
  const lines = [];
  const stale = row.connection && row.connection !== 'online' || snapshot.generated_at - row.updated_at > 90;
  const fallback = stale ? '数据待刷新' : '等待实测进度';
  const trainBasis = '按近期实测训练步速估算。';
  if (current) {
    const checkpoint = row.checkpoint_step;
    if (stages.length === 1 && Number.isFinite(checkpoint) && current.current < checkpoint &&
        Number.isFinite(current.current) && checkpoint < current.target) {
      lines.push(completionLine(`${num(checkpoint)} 步预计到达`, estimates.checkpoint, fallback, trainBasis));
    }
    const label = stages.length > 1 ? '当前阶段预计完成' : '训练预计完成';
    const pending = current.current >= current.target ? '目标步数已到，等待确认' : fallback;
    lines.push(completionLine(label, estimates.training, pending, trainBasis));
  }
  const states = {queued:'排队中，等待评测启动', confirming:'场次已跑完，等待审核',
    waiting:'尚未启动', complete:'已完成', trained:'训练已完成', stopped:'已停止', failed:'失败',
    error:'异常，待核实', stale:'数据待刷新', paused:'已暂停'};
  if (row.state === 'testing') {
    lines.push(completionLine('评测预计跑完', estimates.evaluation, fallback,
      '按近期正式场次入账速度估算。并行组数和场次耗时变化会影响估算，审核时间另计。'));
  } else if (states[row.state]) {
    lines.push(`<div><span>完成时间</span> ${states[row.state]}</div>`);
  }
  return lines.length ? `<div class="completion-times">${lines.join('')}</div>` : '';
}

function training(row) {
  const values = (row.training || []).map(s => {
    const current = Number.isFinite(s.current) ? s.current : null;
    const target = s.target > 0 ? s.target : null;
    const amount = target && current !== null ? Math.min(100, 100 * current / target) : 0;
    const label = current === null ? STATES[s.state] || '未登记' : `${num(current)}${target ? ` / ${num(target)}` : ' 步'}`;
    return `<div class="train-stage"><div class="stage-top"><span>${escapeHTML(s.label)}</span><span>${label}</span></div>
      <progress class="progress ${s.state === 'complete' ? 'complete' : ''}" max="100" value="${amount}"
        aria-label="${escapeHTML(s.label)}">${amount}%</progress></div>`;
  }).join('');
  const completed = (row.groups || []).reduce((n, g) => n + (g.completed || 0), 0);
  const accepted = row.groups?.some(group => Number.isInteger(group.accepted)) ?
    row.groups.reduce((n, group) => n + group.accepted, 0) : completed;
  const evalText = accepted < completed ? `已跑完 ${num(completed)} / ${num(row.expected)} 场 · 正式入账 ${num(accepted)} 场` :
    completed ? `测试 ${completed}${row.expected ? ` / ${row.expected}` : ''} 次` :
    row.expected ? `测试 0 / ${row.expected} 次` : '尚无正式测试';
  const active = (row.workers || []).filter(w => w.live !== false);
  const activeCount = row.live_groups ?? active.length;
  return `${values || '<span class="na">未关联训练记录</span>'}<div class="sub">${evalText}</div>
    ${activeCount ? `<div class="execution">${activeCount} 组执行中${active.length && active.every(w => w.gpu !== undefined) ?
      ' · GPU ' + [...new Set(active.map(w => w.gpu))].join('/') : ''}</div>` : ''}
    ${renderCompletion(row)}`;
}

function updateFilter(id, allLabel, options) {
  const select = $(id), value = select.value;
  select.innerHTML = `<option value="">${allLabel}</option>` + [...options].map(([key, name]) =>
    `<option value="${escapeHTML(key)}">${escapeHTML(name)}</option>`).join('');
  select.value = options.has(value) ? value : '';
  return select.value;
}

function matches(row, {task, method, family, state, query}) {
  if (task && task !== row.taskKey || method && method !== row.methodName ||
      family && family !== row.family) return false;
  if (state === 'active' && !['testing','training','running'].includes(row.state)) return false;
  if (state && state !== 'active' && state !== row.state) return false;
  return !query || [row.title, row.methodName, row.family, row.taskName, row.host, row.root, row.note, ...row.visual]
    .join(' ').toLocaleLowerCase().includes(query);
}

function renderVisual([encoder, alignment, training]) {
  return '<dl class="visual-profile">' + [
    ['视觉表征', encoder], ['换臂预对齐', alignment], ['动作训练时', training],
  ].map(([label, value]) => `<dt>${label}</dt><dd>${escapeHTML(value)}</dd>`).join('') + '</dl>';
}

function renderRow(row) {
  return `<tr><td><button class="method" data-id="${escapeHTML(row.id)}">${escapeHTML(row.title)}</button>
    ${renderVisual(row.visual)}
    ${row.state === 'stopped' && row.id.startsWith('aghost_franka100_hang_') && row.note ?
      `<div class="sub">${escapeHTML(row.note)}</div>` : ''}
    <div class="sub">${escapeHTML(row.family)}${row.host ? ' · ' + escapeHTML(row.host) + ' 服务器' : ''}</div><div class="badges">
    <span class="badge ${escapeHTML(row.state)}">${escapeHTML(STATES[row.state] || row.state || '未登记')}</span>
    ${row.audited ? '<span class="badge complete">证据已审计</span>' : ''}
    ${row.connection && row.connection !== 'online' ? '<span class="badge stale">数据过期</span>' : ''}</div>
    ${editing ? `<button class="delete-record" data-record-action="remove" data-id="${escapeHTML(row.id)}"
      aria-label="删除 ${escapeHTML(row.title)}">删除</button>` : ''}</td>
    <td>${training(row)}</td>${cell(row,'总计')}${BODIES.map(b=>cell(row,b)).join('')}</tr>`;
}

function renderTask(taskName, rows) {
  const methods = new Map();
  for (const row of rows) {
    if (!methods.has(row.methodName)) methods.set(row.methodName, []);
    methods.get(row.methodName).push(row);
  }
  return `<section class="task-section"><header class="task-head"><h3>${escapeHTML(taskName)}</h3>
    <small>${methods.size} 种方法 · ${rows.length} 条记录</small></header>
    <div class="scroll"><table aria-label="${escapeHTML(taskName)}训练与测试记录"><thead><tr>
      <th scope="col">实验记录</th><th scope="col">训练与测试进度 <small class="eta-zone">预计完成时间 · 北京时间</small></th>
      <th scope="col" class="body-head total-column">总计 · 五机械臂</th>${BODIES.map(body =>
        `<th scope="col" class="body-head">${body}</th>`).join('')}</tr></thead>
    ${[...methods].map(([name, records]) => `<tbody><tr class="method-heading">
      <th scope="rowgroup" colspan="${3 + BODIES.length}"><div><h4>${escapeHTML(name)}</h4>
      <small>${records.length} 条记录</small></div></th></tr>
      ${records.map(renderRow).join('')}</tbody>`).join('')}</table></div></section>`;
}

function render() {
  if (!snapshot) return;
  const active = document.activeElement;
  const action = active?.dataset.recordAction;
  const actionSelector = action && `button[data-record-action="${action}"]`;
  const actionIndex = action && [...document.querySelectorAll(actionSelector)].indexOf(active);
  const classified = snapshot.rows.map(classifyRow);
  const rows = classified.filter(row => !removedRecords.has(row.id));
  const dialog = $('detail');
  if (dialog.open) {
    const selected = rows.find(row => row.id === dialog.dataset.id);
    if (selected) {
      renderDetail(selected);
    } else {
      dialog.close();
      $('detail-content').replaceChildren();
    }
  }
  const tasks = new Map([...TASK_NAMES, ...classified.map(row => [row.taskKey, row.taskName])]);
  const task = updateFilter('task', '全部任务', tasks);
  const taskRows = task ? classified.filter(row => row.taskKey === task) : classified;
  const method = updateFilter('method-filter', '全部方法',
    new Map(taskRows.map(row => [row.methodName, row.methodName])));
  const methodRows = method ? taskRows.filter(row => row.methodName === method) : taskRows;
  const family = updateFilter('family', '全部对比组', new Map(methodRows.map(row => [row.family, row.family])));
  $('records').textContent = num(rows.length);
  $('methods').textContent = num(new Set(rows.map(row => row.methodName)).size);
  $('training').textContent = num(rows.filter(row =>
    ['training','running'].includes(row.state) && row.training?.length).length);
  $('testing').textContent = num(rows.filter(row => row.state === 'testing').length);
  const feeds = Object.entries(snapshot.feeds || {}), bad = feeds.filter(([,f]) => f.error || f.age_seconds > 90);
  const ready = feeds.length >= 2;
  if (snapshotExpired() && !snapshot.feeds?.['公网同步']) {
    bad.push(['公网同步', {error:'快照过期'}]);
  }
  $('live-status').textContent = bad.length ? '部分数据过期' : ready ? '实时数据已连接' : '正在连接数据源';
  $('updated').textContent = `页面更新 ${new Date(snapshot.generated_at * 1000).toLocaleTimeString('zh-CN',{hour12:false})} · ${refreshSeconds}秒刷新`;
  $('feed-alert').hidden = !bad.length;
  $('feed-alert').textContent = bad.length ? '数据源暂时不可用，保留上次成功读取的结果：' +
    bad.map(([name,f]) => `${name}（${f.error || '超过90秒未更新'}）`).join('、') : '';
  if (refreshError) {
    $('live-status').textContent = '连接中断，保留已有结果';
    $('feed-alert').hidden = false;
    $('feed-alert').textContent = '网页数据刷新失败：' + refreshError;
  }
  const filters = {task, method, family, state: $('state').value, query: $('search').value.trim().toLocaleLowerCase()};
  const visible = rows.filter(row => matches(row, filters)), grouped = new Map();
  renderTotals(visible);
  for (const row of visible) {
    if (!grouped.has(row.taskKey)) grouped.set(row.taskKey, []);
    grouped.get(row.taskKey).push(row);
  }
  $('visible-count').textContent = `显示 ${visible.length} / ${rows.length} 条记录`;
  renderRecordEditor();
  $('experiments').innerHTML = [...tasks].filter(([key]) => grouped.has(key))
    .map(([key, name]) => renderTask(name, grouped.get(key))).join('') ||
    '<div class="empty">没有符合筛选条件的实验记录。</div>';
  if (action) {
    const buttons = [...document.querySelectorAll(actionSelector)];
    const next = buttons.find(button => button.dataset.id === active.dataset.id) ||
      buttons[Math.min(actionIndex, buttons.length - 1)] || $('edit-records');
    next.focus({preventScroll:true});
  }
}

function renderDetail(row) {
  $('detail-content').innerHTML = `<div class="eyebrow">EXPERIMENT DETAIL</div><h2>${escapeHTML(row.title)}</h2>
    <p>${escapeHTML(row.taskName)} · ${escapeHTML(row.methodName)}</p>
    <p>${escapeHTML(row.family)}</p>${renderVisual(row.visual)}<div id="detail-completion">${renderCompletion(row)}</div>
    ${row.note ? `<h4>实验说明</h4><p>${escapeHTML(row.note)}</p>` : ''}
    ${row.notes ? `<p>${escapeHTML(Array.isArray(row.notes) ? row.notes.join('；') : row.notes)}</p>` : ''}
    ${snapshot.public_view ? `<p>记录更新时间 ${escapeHTML(time(row.updated_at))}</p>` : `
    <h4>结果来源</h4><pre>${escapeHTML(row.evidence || row.provenance || row.root || '历史实验比较表')}
${escapeHTML(row.root || '')}
记录更新时间：${escapeHTML(time(row.updated_at))}</pre>
    <h4>正在执行</h4><pre>${escapeHTML(row.workers?.length ? JSON.stringify(row.workers,null,2) : '当前无执行组 / 未关联执行记录')}</pre>`}
    <h4>逐机械臂计数</h4><pre>${escapeHTML(JSON.stringify(row.groups,null,2))}</pre>
    ${row.accounting ? `<h4>统计状态</h4><pre>${escapeHTML(JSON.stringify(row.accounting,null,2))}</pre>` : ''}`;
}

function detail(id) {
  const record = snapshot.rows.find(row => row.id === id);
  if (!record) return;
  renderDetail(classifyRow(record));
  $('detail').dataset.id = id;
  $('detail').showModal();
}

function snapshotExpired() {
  return Boolean(dashboardConfig.snapshotUrl && snapshot &&
    Date.now() / 1000 - snapshot.generated_at > refreshSeconds * 4);
}

async function fetchPublicSnapshot(url, headers) {
  try {
    const response = await fetch(url, {cache:'no-store', credentials:'omit', headers,
      signal:AbortSignal.timeout(4500)});
    if (!response.ok) return null;
    const data = await response.json();
    return data?.public_view === true && Number.isFinite(data.generated_at) &&
      Array.isArray(data.rows) && data.rows.length ? data : null;
  } catch {
    return null;
  }
}

async function loadSnapshot() {
  const live = await fetchPublicSnapshot(dashboardConfig.liveSnapshotUrl);
  const now = Date.now();
  let latest = live;
  if (live && now / 1000 - live.generated_at <= 45 &&
      live.generated_at >= (snapshot?.generated_at ?? 0)) return live;

  const url = new URL(dashboardConfig.snapshotUrl);
  url.searchParams.set('t', Math.floor(now / 30000));
  const raw = await fetchPublicSnapshot(url);
  if (raw && (!latest || raw.generated_at > latest.generated_at)) latest = raw;

  const newestAt = Math.max(latest?.generated_at ?? 0, snapshot?.generated_at ?? 0);
  if (now / 1000 - newestAt > 45 && now - lastApiRefresh >= 90000) {
    lastApiRefresh = now;
    const data = await fetchPublicSnapshot(dashboardConfig.snapshotApi,
      {Accept:'application/vnd.github.raw+json'});
    if (data && (!latest || data.generated_at > latest.generated_at)) latest = data;
  }
  if (!latest) throw new Error('暂时无法读取实验快照');
  return latest;
}

async function refresh() {
  if (fetching) return;
  fetching = true;
  try {
    const next = await loadSnapshot();
    if (!snapshot || next.generated_at >= snapshot.generated_at) snapshot = next;
    refreshError = null;
    render();
  } catch (error) {
    refreshError = error.message;
    if (snapshot) render();
    else {
      $('live-status').textContent = '连接中断';
      $('feed-alert').hidden = false;
      $('feed-alert').textContent = '网页数据刷新失败：' + refreshError;
    }
  } finally { fetching = false; }
}

if (typeof document !== 'undefined') {
  removedRecords = readRemovedRecords();
  window.addEventListener('storage', event => {
    if (event.key === REMOVED_RECORDS_KEY || event.key === null) {
      removedRecords = readRemovedRecords();
      render();
    }
  });
  ['task','method-filter','family','state'].forEach(id => $(id).addEventListener('change',render));
  $('search').addEventListener('input',render);
  $('refresh').addEventListener('click',refresh);
  $('close-detail').addEventListener('click',()=> $('detail').close());
  $('edit-records').addEventListener('click', () => {
    editing = !editing;
    $('edit-status').textContent = '';
    render();
  });
  $('experiments').addEventListener('click', event => {
    const button = event.target.closest('button[data-id]');
    if (!button) return;
    if (button.dataset.recordAction === 'remove') setRecordRemoved(button.dataset.id, true);
    else detail(button.dataset.id);
  });
  $('removed-list').addEventListener('click', event => {
    const button = event.target.closest('button[data-record-action="restore"]');
    if (button) setRecordRemoved(button.dataset.id, false);
  });
  document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click',()=> {
    selectedMetric=button.dataset.metric;
    document.querySelectorAll('[data-metric]').forEach(b=> { b.classList.toggle('selected',b===button); b.setAttribute('aria-pressed',String(b===button)); });
    render();
  }));
  refresh(); setInterval(refresh,refreshSeconds * 1000);
}
