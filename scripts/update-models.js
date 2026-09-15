#!/usr/bin/env node
/**
 * Pareto Inference model catalog sync.
 *
 * Fetches the public model catalog from https://api.paretoinference.com/v1/models,
 * writes models.json, moves delisted models into deprecated-models.json, and
 * regenerates the README model table.
 *
 * GET /v1/models does not require an API key. PARETO_API_KEY is still sent when
 * present so this keeps working if Pareto ever gates the catalog.
 *
 * Usage:
 *   node scripts/update-models.js
 *   PARETO_API_KEY=your-key node scripts/update-models.js
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MODELS_JSON_PATH = path.join(PROJECT_ROOT, 'models.json');
const README_PATH = path.join(PROJECT_ROOT, 'README.md');
const PATCH_JSON_PATH = path.join(PROJECT_ROOT, 'patch.json');
const CUSTOM_MODELS_JSON_PATH = path.join(PROJECT_ROOT, 'custom-models.json');

const ENDPOINT_ROOT = (process.env.PARETO_BASE_URL || 'https://api.paretoinference.com/v1').replace(/\/+$/, '');
const MODELS_API_URL = `${ENDPOINT_ROOT}/models`;

// Pareto documents max_tokens 1–131,072 but has not published deployment
// context/output limits; conservative floors, overridable via patch.json.
const CONSERVATIVE_CONTEXT_WINDOW = 131072;
const CONSERVATIVE_MAX_TOKENS = 131072;

function loadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function saveJson(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n');
}

// Patch application (same pipeline as index.ts)

function applyPatch(model, patch) {
  const result = { ...model };
  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = { ...patch.thinkingLevelMap };
  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }
  if (!result.reasoning && result.compat?.thinkingFormat) {
    delete result.compat.thinkingFormat;
  }
  if (!result.reasoning && result.thinkingLevelMap) {
    delete result.thinkingLevelMap;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }
  return result;
}

function buildModels(baseModels, customModels, patchData) {
  const modelMap = new Map();
  for (const model of baseModels) modelMap.set(model.id, model);
  for (const [id, patchEntry] of Object.entries(patchData)) {
    const existing = modelMap.get(id);
    if (existing) modelMap.set(id, applyPatch(existing, patchEntry));
  }
  for (const model of customModels) {
    const existing = modelMap.get(model.id);
    const patchEntry = patchData[model.id];
    if (existing && patchEntry) modelMap.set(model.id, applyPatch(model, patchEntry));
    else if (existing) modelMap.set(model.id, model);
    else if (patchEntry) modelMap.set(model.id, applyPatch(model, patchEntry));
    else modelMap.set(model.id, model);
  }
  return Array.from(modelMap.values());
}

// Model transformation

// pi thinking levels → reasoning_effort values the Pareto API accepts.
// `xhigh` is rejected with HTTP 500, so it stays null (hidden).
const THINKING_LEVEL_MAP = {
  off: 'none',
  minimal: 'minimal',
  low: 'low',
  medium: 'medium',
  high: 'high',
  xhigh: null,
  max: 'max',
};

function generateDisplayName(id) {
  const raw = id.includes('/') ? id.slice(id.indexOf('/') + 1) : id;
  const name = raw
    .split(/[-_]/)
    .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
    .join(' ');
  return name.replace(/^Deepseek/, 'DeepSeek').replace(/^Glm/, 'GLM');
}

function transformModel(apiModel) {
  return {
    id: apiModel.id,
    name: generateDisplayName(apiModel.id),
    reasoning: true,
    input: ['text'],
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    },
    contextWindow: apiModel.context_window || CONSERVATIVE_CONTEXT_WINDOW,
    maxTokens: apiModel.max_tokens || CONSERVATIVE_MAX_TOKENS,
    thinkingLevelMap: { ...THINKING_LEVEL_MAP },
    compat: {
      thinkingFormat: 'openai',
      supportsReasoningEffort: true,
      supportsDeveloperRole: false,
      supportsStore: false,
      maxTokensField: 'max_tokens',
    },
  };
}

// README generation

function formatCost(cost) {
  if (cost === 0 || cost === null || cost === undefined) return '—';
  return '$' + cost.toFixed(2);
}

function formatNumber(num) {
  if (num === null || num === undefined) return '-';
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${Math.round(num / 1000)}K`;
  return num.toString();
}

function getInputTypes(inputTypes) {
  const types = inputTypes || ['text'];
  if (types.includes('image') && types.includes('text')) return 'Text + Image';
  if (types.includes('image')) return 'Image';
  return 'Text';
}

function generateReadmeRow(model) {
  const cost = model.cost || {};
  return `| ${model.name} | ${getInputTypes(model.input)} | ${formatNumber(model.contextWindow)} | ${formatNumber(model.maxTokens)} | ${formatCost(cost.input)} | ${formatCost(cost.output)} |`;
}

function updateReadme(models) {
  let readme = fs.readFileSync(README_PATH, 'utf8');

  const sortedModels = [...models].sort((a, b) => a.name.localeCompare(b.name));
  const tableRows = sortedModels.map(generateReadmeRow).join('\n');
  const newTable = `| Model | Type | Context | Max Tokens | Input Cost | Output Cost |
|-------|------|---------|------------|------------|-------------|
${tableRows}`;

  const tableRegex = /\| Model \| Type \| Context \| Max Tokens \| Input Cost \| Output Cost \|[\s\S]*?(?=\n\*Costs are per million)/;
  readme = readme.replace(tableRegex, newTable);

  readme = readme.replace(/\*\*\d+\+ AI Models\*\*/, `**${models.length}+ AI Models**`);

  fs.writeFileSync(README_PATH, readme);
  console.log(`✓ Updated README.md with ${models.length} models`);
}

// Deprecated-model reconciliation

const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Reconcile deprecated-models.json against the freshly fetched model list.
 * - in old models.json but not the API: moved into the deprecated file
 *   (deprecatedAt = now; preserved on repeat runs so the grace clock is not reset)
 * - back in the API: resurrected (dropped from the deprecated file)
 * - deprecatedAt older than 14 days: evicted permanently
 * Must run BEFORE the new models.json is written; it reads the old file itself.
 */
function updateDeprecatedModels(modelsJsonPath, newModels) {
  const deprecatedPath = path.join(path.dirname(modelsJsonPath), 'deprecated-models.json');

  let oldModels = [];
  try {
    const parsed = JSON.parse(fs.readFileSync(modelsJsonPath, 'utf8'));
    if (Array.isArray(parsed)) oldModels = parsed;
  } catch { /* first run: no previous models.json */ }

  let deprecated = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(deprecatedPath, 'utf8'));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) deprecated = parsed;
  } catch { /* no graveyard yet */ }

  const currentIds = new Set(newModels.map(m => m.id));
  const now = new Date().toISOString();
  const added = [];
  const resurrected = [];
  const evicted = [];

  for (const old of oldModels) {
    if (old && old.id && !currentIds.has(old.id) && !deprecated[old.id]) {
      deprecated[old.id] = { ...old, deprecatedAt: now };
      added.push(old.id);
    }
  }

  for (const [id, entry] of Object.entries(deprecated)) {
    if (currentIds.has(id)) {
      delete deprecated[id];
      resurrected.push(id);
      continue;
    }
    const removedAt = Date.parse(entry && entry.deprecatedAt ? entry.deprecatedAt : '');
    if (Number.isNaN(removedAt) || Date.now() - removedAt > DEPRECATED_MODEL_TTL_MS) {
      delete deprecated[id];
      evicted.push(id);
    }
  }

  if (added.length > 0 || resurrected.length > 0 || evicted.length > 0) {
    fs.writeFileSync(deprecatedPath, JSON.stringify(deprecated, null, 2) + '\n');
    console.log('Updated deprecated-models.json ' + JSON.stringify({ added, resurrected, evicted }));
  }
}

/**
 * Grace-period deprecated models (deprecatedAt within TTL) with metadata stripped.
 * Keeps the README table serving models that are delisted but still within their
 * 14-day grace window.
 */
function withDeprecatedForReadme(models) {
  const deprecatedPath = path.join(PROJECT_ROOT, 'deprecated-models.json');
  let deprecated = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(deprecatedPath, 'utf8'));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) deprecated = parsed;
  } catch { /* no graveyard yet */ }
  const now = Date.now();
  const seen = new Set(models.map(m => m.id));
  const extras = [];
  for (const entry of Object.values(deprecated)) {
    if (!entry || !entry.id || seen.has(entry.id)) continue;
    const removedAt = Date.parse(entry.deprecatedAt || '');
    if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
    const m = { ...entry };
    delete m.deprecatedAt;
    extras.push(m);
  }
  return extras.length > 0 ? [...models, ...extras] : models;
}

// Main

async function main() {
  console.log(`Fetching models from ${MODELS_API_URL}...`);

  try {
    const headers = {};
    if (process.env.PARETO_API_KEY) headers.Authorization = `Bearer ${process.env.PARETO_API_KEY}`;
    const response = await fetch(MODELS_API_URL, { headers });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const apiResponse = await response.json();
    const apiModels = Array.isArray(apiResponse)
      ? apiResponse
      : (apiResponse.data || apiResponse.models || []);

    if (!Array.isArray(apiModels)) {
      throw new Error('API response does not contain an array of models');
    }

    console.log(`✓ Fetched ${apiModels.length} models from API`);

    // Load existing models.json — source of truth for curated specs
    let existingModels = [];
    try {
      existingModels = JSON.parse(fs.readFileSync(MODELS_JSON_PATH, 'utf8'));
    } catch {
      // File might not exist yet
    }
    const existingModelsMap = {};
    for (const m of existingModels) {
      existingModelsMap[m.id] = m;
    }

    // Transform models from API. The live catalog is authoritative for
    // context/output limits; curated name/thinkingLevelMap/compat wins.
    const apiTransformed = apiModels
      .map(m => ({ ...transformModel(m), ...(existingModelsMap[m.id] ?? {}) }))
      .sort((a, b) => a.name.localeCompare(b.name));

    // Load patch overrides for README rendering.
    const patch = loadJson(PATCH_JSON_PATH);

    // Update models.json — curated API data
    // Move delisted models to deprecated-models.json BEFORE models.json is overwritten
    updateDeprecatedModels(MODELS_JSON_PATH, apiTransformed);
    fs.writeFileSync(MODELS_JSON_PATH, JSON.stringify(apiTransformed, null, 2) + '\n');
    console.log(`✓ Updated models.json (${apiTransformed.length} models)`);

    // Load custom-models.json
    const customModels = Array.isArray(loadJson(CUSTOM_MODELS_JSON_PATH))
      ? loadJson(CUSTOM_MODELS_JSON_PATH)
      : [];

    // Check for custom models now available upstream (remove duplicates)
    const upstreamIds = new Set(apiTransformed.map(m => m.id));
    const duplicates = customModels.filter(m => upstreamIds.has(m.id));
    if (duplicates.length > 0) {
      console.log(`\nFound ${duplicates.length} custom model(s) now available upstream:`);
      for (const dup of duplicates) {
        console.log(`  - ${dup.id} (${dup.name})`);
      }
      const cleaned = customModels.filter(m => !upstreamIds.has(m.id));
      saveJson(CUSTOM_MODELS_JSON_PATH, cleaned);
      console.log(`✓ Removed ${duplicates.length} duplicate(s) from custom-models.json`);
      customModels.length = 0;
      customModels.push(...cleaned);
    }

    // Build merged models with patches for README
    const readmeModels = buildModels(withDeprecatedForReadme(apiTransformed), customModels, patch);
    readmeModels.sort((a, b) => a.name.localeCompare(b.name));

    // Update README
    updateReadme(readmeModels);

    // Summary
    console.log('\n--- Summary ---');
    console.log(`Total models: ${readmeModels.length}`);
    console.log(`Reasoning models: ${readmeModels.filter(m => m.reasoning).length}`);
    console.log(`Vision models: ${readmeModels.filter(m => m.input.includes('image')).length}`);

    const newIds = new Set(apiTransformed.map(m => m.id));
    const oldIds = new Set(existingModels.map(m => m.id));

    const added = [...newIds].filter(id => !oldIds.has(id));
    const removed = [...oldIds].filter(id => !newIds.has(id));

    if (added.length > 0) console.log(`\nNew models: ${added.join(', ')}`);
    if (removed.length > 0) console.log(`\nRemoved models: ${removed.join(', ')}`);

    console.log('\nDone!');
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
