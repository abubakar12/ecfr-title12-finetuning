export type ModelConfig = { id: string; name: string; stage: "base" | "cpt" | "sft" | "dpo" | "grpo" | "custom"; modelId: string; endpointUrl?: string; color: "slate" | "cyan" | "violet" | "amber" | "rose" };
export const DEFAULT_MODELS: ModelConfig[] = [
  { id: "base", name: "Base · GPU", stage: "base", modelId: "meta-llama/Meta-Llama-3.1-8B-Instruct", endpointUrl: "local://base", color: "slate" },
  { id: "cpt", name: "CPT · GPU", stage: "cpt", modelId: "abubakarilyas624/ecfr-title12-cpt-e0b4a43ee1aa", endpointUrl: "local://cpt", color: "cyan" },
  { id: "sft", name: "SFT · GPU", stage: "sft", modelId: "abubakarilyas624/ecfr-title12-sft-3f8296763816", endpointUrl: "local://sft", color: "violet" },
  { id: "dpo", name: "DPO · GPU", stage: "dpo", modelId: "abubakarilyas624/ecfr-title12-dpo-d361734506db", endpointUrl: "local://dpo", color: "amber" },
  { id: "grpo", name: "GRPO · GPU", stage: "grpo", modelId: "abubakarilyas624/ecfr-title12-grpo-665d8550958a", endpointUrl: "local://grpo", color: "rose" },
];
export const SYSTEM_PROMPT = "You are a banking-regulation assistant specialized in Title 12 of the Code of Federal Regulations. Answer accurately, ground your answer in the regulation text, and cite the relevant section.";
