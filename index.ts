/**
 * Pareto Inference Provider Extension
 *
 * Registers Pareto Inference (paretoinference.com) as a custom provider.
 * Base URL: https://api.paretoinference.com/v1 (OpenAI Chat Completions).
 *
 * Pareto serves GLM 5.3 Flash on its own GPUs; its public catalog also lists
 * GLM 5.3 and DeepSeek V4 Flash. Reasoning arrives through `reasoning_effort`,
 * tool calls follow the OpenAI schema, and streaming uses SSE with
 * `stream_options: { include_usage: true }`.
 *
 * Model resolution strategy: Stale-While-Revalidate
 *   1. Serve stale immediately: disk cache → embedded models.json (zero-latency)
 *   2. Revalidate in background: GET /v1/models (public, no key) → merge with
 *      embedded → cache → hot-swap
 *   3. patch.json + custom-models.json applied on top of whichever source won
 *
 * Merge order: [live|cache|embedded] → apply patch.json → merge custom-models.json
 *
 * The catalog endpoint is public, so model sync never blocks on credentials;
 * only chat completions need an API key.
 *
 * Endpoint behavior verified against the live API (2026-09, z-ai/glm-5.3-flash):
 *   - reasoning_effort accepts none | minimal | low | medium | high | max;
 *     xhigh returns HTTP 500, so it is hidden via thinkingLevelMap
 *   - image content parts are accepted (text + image_url)
 *   - replayed assistant tool calls without reasoning_content are accepted
 *   - store: true is rejected (HTTP 400 unsupported_storage) → supportsStore: false
 *   - max_tokens and max_completion_tokens are both accepted, up to 131,072
 *
 * Pareto has not published context/output deployment limits, so the embedded
 * catalog carries a conservative 131,072-token context and output floor derived
 * from the documented max_tokens range (1–131,072). Raise them in patch.json
 * once Pareto publishes real deployment limits.
 *
 * Usage:
 *   # Option 1: Store in auth.json (recommended)
 *   # Add to ~/.pi/agent/auth.json:
 *   #   "pareto": { "type": "api_key", "key": "your-api-key" }
 *
 *   # Option 2: Set as environment variable
 *   export PARETO_API_KEY=your-api-key
 *
 *   # Run pi with the extension
 *   pi -e /path/to/pi-pareto-inference-provider
 *
 * Then use /model to select from available models.
 *
 * @see https://docs.paretoinference.com
 */

import { getAgentDir, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };
import deprecatedData from "./deprecated-models.json" with { type: "json" };
import fs from "fs";
import path from "path";

// Types

interface JsonModel {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
	maxTokens: number;
	thinkingLevelMap?: Record<string, string | null>;
	compat?: {
		supportsDeveloperRole?: boolean;
		supportsStore?: boolean;
		maxTokensField?: "max_completion_tokens" | "max_tokens";
		thinkingFormat?: "openai" | "zai" | "qwen" | "qwen-chat-template" | "deepseek" | "openrouter" | "baseten" | "chat-template";
		supportsReasoningEffort?: boolean;
		requiresReasoningContentOnAssistantMessages?: boolean;
	};
}

interface PatchEntry {
	name?: string;
	reasoning?: boolean;
	input?: ("text" | "image")[];
	cost?: {
		input?: number;
		output?: number;
		cacheRead?: number;
		cacheWrite?: number;
	};
	contextWindow?: number;
	maxTokens?: number;
	thinkingLevelMap?: Record<string, string | null>;
	compat?: Record<string, unknown>;
}

type PatchData = Record<string, PatchEntry>;

// Patch application

function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
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

/** Full pipeline: base models → patch → custom → result */
function buildModels(base: JsonModel[], custom: JsonModel[], patch: PatchData): JsonModel[] {
	const modelMap = new Map<string, JsonModel>();

	// Seed with the base list plus grace-period deprecated models so patch.json
	// entries apply to deprecated models exactly as while the model was live
	// (withDeprecated keeps live data on id conflicts).
	for (const model of withDeprecated(base)) {
		modelMap.set(model.id, model);
	}

	for (const [id, patchEntry] of Object.entries(patch)) {
		const existing = modelMap.get(id);
		if (existing) {
			modelMap.set(id, applyPatch(existing, patchEntry));
		}
	}

	for (const model of custom) {
		const existing = modelMap.get(model.id);
		const patchEntry = patch[model.id];
		if (existing && patchEntry) {
			modelMap.set(model.id, applyPatch(model, patchEntry));
		} else if (existing) {
			modelMap.set(model.id, model);
		} else if (patchEntry) {
			modelMap.set(model.id, applyPatch(model, patchEntry));
		} else {
			modelMap.set(model.id, model);
		}
	}

	return Array.from(modelMap.values());
}

// Stale-While-Revalidate model sync

const PROVIDER_ID = "pareto";
// Endpoint root is overridable for proxies and tests.
const BASE_URL = (process.env.PARETO_BASE_URL || "https://api.paretoinference.com/v1").replace(/\/+$/, "");
const MODELS_URL = `${BASE_URL}/models`;
const CACHE_DIR = path.join(getAgentDir(), "cache");
const CACHE_PATH = path.join(CACHE_DIR, `${PROVIDER_ID}-models.json`);
const LIVE_FETCH_TIMEOUT_MS = 8000;

// Pareto documents max_tokens 1–131,072 but has not published deployment
// context/output limits. These conservative floors keep pi's auto-compaction
// honest until real limits are published; patch.json is the override point.
const CONSERVATIVE_CONTEXT_WINDOW = 131072;
const CONSERVATIVE_MAX_TOKENS = 131072;

// pi thinking levels → reasoning_effort values the Pareto API accepts.
// `xhigh` is rejected with HTTP 500, so it stays null (hidden).
const THINKING_LEVEL_MAP: Record<string, string | null> = {
	off: "none",
	minimal: "minimal",
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: null,
	max: "max",
};

function generateDisplayName(id: string): string {
	const raw = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
	const name = raw
		.split(/[-_]/)
		.map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
		.join(" ");
	return name.replace(/^Deepseek/, "DeepSeek").replace(/^Glm/, "GLM");
}

/** Transform a model from Pareto's public /v1/models catalog. */
function transformApiModel(apiModel: any): JsonModel | null {
	if (typeof apiModel?.id !== "string" || apiModel.id.length === 0) return null;

	return {
		id: apiModel.id,
		name: generateDisplayName(apiModel.id),
		reasoning: true,
		input: ["text"],
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
			thinkingFormat: "openai",
			supportsReasoningEffort: true,
			supportsDeveloperRole: false,
			supportsStore: false,
			maxTokensField: "max_tokens",
		},
	};
}

async function fetchLiveModels(signal?: AbortSignal): Promise<JsonModel[] | null> {
	try {
		// The catalog is public; the key is optional and only sent when present.
		const headers: Record<string, string> = {};
		const apiKey = process.env.PARETO_API_KEY;
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		const response = await fetch(MODELS_URL, {
			headers,
			signal: signal ? AbortSignal.any([AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS), signal]) : AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS),
		});
		if (!response.ok) return null;
		const data = await response.json();
		const apiModels = Array.isArray(data) ? data : (data.data || []);
		if (!Array.isArray(apiModels) || apiModels.length === 0) return null;
		return apiModels.map(transformApiModel).filter((m): m is JsonModel => m !== null);
	} catch {
		return null;
	}
}

function loadCachedModels(): JsonModel[] | null {
	try {
		const data = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
		return Array.isArray(data) ? data : null;
	} catch {
		return null;
	}
}

function cacheModels(models: JsonModel[]): void {
	try {
		fs.mkdirSync(CACHE_DIR, { recursive: true });
		fs.writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2) + "\n");
	} catch {
		// Cache write failure is non-fatal
	}
}

function mergeWithEmbedded(liveModels: JsonModel[], embeddedModels: JsonModel[]): JsonModel[] {
	const embeddedMap = new Map(embeddedModels.map(m => [m.id, m]));
	const seen = new Set<string>();
	const result: JsonModel[] = [];
	for (const liveModel of liveModels) {
		const embedded = embeddedMap.get(liveModel.id);
		seen.add(liveModel.id);
		if (embedded) {
			// /v1/models reports ids only today, so curation wins; if Pareto starts
			// reporting context/output limits, the live values win for those fields.
			result.push({
				...liveModel,
				...embedded,
				contextWindow: liveModel.contextWindow || embedded.contextWindow,
				maxTokens: liveModel.maxTokens || embedded.maxTokens,
			});
		} else {
			result.push(liveModel);
		}
	}
	// Append any embedded models that the live API didn't return
	for (const em of embeddedModels) {
		if (!seen.has(em.id)) {
			result.push(em);
		}
	}
	return result;
}

// Grace period for delisted models. When the provider API stops listing a
// model, update-models.js moves its last-known definition into
// deprecated-models.json (stamped with deprecatedAt) instead of dropping it.
// For 14 days the model keeps working here so in-flight sessions and saved
// model settings do not break; afterwards it is evicted permanently.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

// Grace-period deprecated models with deprecation metadata stripped.
function activeDeprecatedModels(): JsonModel[] {
	const now = Date.now();
	const result: JsonModel[] = [];
	for (const entry of Object.values(deprecatedData as Record<string, JsonModel & { deprecatedAt?: string }>)) {
		if (!entry?.id) continue;
		const removedAt = Date.parse(entry.deprecatedAt ?? "");
		if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
		const model = { ...entry } as JsonModel & { deprecatedAt?: string };
		delete model.deprecatedAt;
		result.push(model);
	}
	return result;
}

// Append grace-period deprecated models the list does not already have (live data wins).
function withDeprecated(models: JsonModel[]): JsonModel[] {
	const seen = new Set(models.map((m) => m.id));
	const extras = activeDeprecatedModels().filter((m) => !seen.has(m.id));
	return extras.length > 0 ? [...models, ...extras] : models;
}

function loadStaleModels(embeddedModels: JsonModel[]): JsonModel[] {
	const cached = loadCachedModels();
	if (!cached || cached.length === 0) return embeddedModels;

	// Merge embedded models that are missing from cache (newly added models)
	const cachedMap = new Map(cached.map(m => [m.id, m]));
	for (const em of embeddedModels) {
		if (!cachedMap.has(em.id)) {
			cached.push(em);
		}
	}
	return cached;
}

async function revalidateModels(embeddedModels: JsonModel[], signal?: AbortSignal): Promise<JsonModel[] | null> {
	const liveModels = await fetchLiveModels(signal);
	if (!liveModels || liveModels.length === 0) return null;
	const merged = mergeWithEmbedded(liveModels, embeddedModels);
	cacheModels(merged);
	return merged;
}

// Extension entry point

// The currently-registered model list — starts stale, hot-swapped when the
// live catalog lands.
let currentModels: JsonModel[] = [];
let revalidateAbort: AbortController | null = null;

function makeProviderConfig(models: JsonModel[] = currentModels) {
	return {
		name: "Pareto Inference",
		baseUrl: BASE_URL,
		apiKey: "$PARETO_API_KEY",
		api: "openai-completions" as const,
		models,
	};
}

export default function (pi: ExtensionAPI) {
	const embeddedModels = modelsData as JsonModel[];
	const customModels = customModelsData as JsonModel[];
	const patches = patchData as PatchData;

	currentModels = buildModels(loadStaleModels(embeddedModels), customModels, patches);
	pi.registerProvider(PROVIDER_ID, makeProviderConfig());

	// Revalidate in the background. GET /v1/models is public, so this never
	// waits on credentials — only chat completions do.
	pi.on("session_start", () => {
		revalidateAbort?.abort();
		revalidateAbort = new AbortController();
		const signal = revalidateAbort.signal;
		revalidateModels(embeddedModels, signal).then((freshBase) => {
			if (freshBase && !signal.aborted) {
				currentModels = buildModels(freshBase, customModels, patches);
				pi.registerProvider(PROVIDER_ID, makeProviderConfig());
			}
		});
	});

	pi.on("session_shutdown", () => {
		revalidateAbort?.abort();
	});
}
