import { readFileSync } from 'node:fs';

/** An explicitly selected local JSON config overrides the bundle's default patch. */
export function loadTrainingConfig(config = {}) {
  const path = config.trainingConfig || process.env.DSH_ADVANCED_CONFIG;
  if (!path) return config;
  let local;
  try { local = JSON.parse(readFileSync(path, 'utf8').replace(/^\uFEFF/, '')); }
  catch (e) { throw new Error(`训练配置读取失败 ${path}: ${e.message}`); }
  if (!local || typeof local !== 'object' || Array.isArray(local)) throw new Error('训练配置必须是 JSON 对象');
  const pipelines = { ...(config.pipelines || {}), ...(local.pipelines || {}) };
  for (const route of ['ddsp', 'gsv', 'rvc', 'anime']) {
    const value = local.pipelines?.[route];
    if (value !== undefined && (!value || typeof value !== 'object' || Array.isArray(value))) throw new Error(`pipelines.${route} 必须是对象`);
    pipelines[route] = { ...(config.pipelines?.[route] || {}), ...(value || {}) };
  }
  return { ...config, ...local, pipelines };
}
