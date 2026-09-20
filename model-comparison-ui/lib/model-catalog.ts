export type ModelConfig = { id: string; name: string; stage: "base" | "cpt" | "sft" | "dpo" | "grpo" | "custom"; modelId: string; endpointUrl?: string; color: "slate" | "cyan" | "violet" | "amber" | "rose" };
export const DEFAULT_MODELS: ModelConfig[] = [
  { id: "base", name: "Base", stage: "base", modelId: "meta-llama/Meta-Llama-3.1-8B-Instruct", color: "slate" },
  { id: "cpt", name: "CPT · complete", stage: "cpt", modelId: "abubakarilyas624/ecfr-title12-cpt-e0b4a43ee1aa", color: "cyan" },
  { id: "sft", name: "SFT · training", stage: "sft", modelId: "", color: "violet" },
  { id: "dpo", name: "DPO · pending", stage: "dpo", modelId: "", color: "amber" },
  { id: "grpo", name: "GRPO · pending", stage: "grpo", modelId: "", color: "rose" },
];
export const SYSTEM_PROMPT = "You are a banking-regulation assistant specialized in Title 12 of the Code of Federal Regulations. Answer accurately, ground your answer in the regulation text, and cite the relevant section.";
