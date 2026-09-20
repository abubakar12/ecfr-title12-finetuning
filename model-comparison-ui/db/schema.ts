import { sql } from "drizzle-orm";
import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const comparisons = sqliteTable("comparisons", {
  id: text("id").primaryKey(), userId: text("user_id").notNull(), title: text("title").notNull(),
  prompt: text("prompt").notNull(), systemPrompt: text("system_prompt").notNull(),
  referenceAnswer: text("reference_answer"), expectedCitation: text("expected_citation"),
  modelConfigs: text("model_configs").notNull(), temperature: integer("temperature_milli").notNull().default(0),
  maxTokens: integer("max_tokens").notNull().default(256), winnerModelId: text("winner_model_id"),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});

export const responses = sqliteTable("responses", {
  id: text("id").primaryKey(), comparisonId: text("comparison_id").notNull().references(() => comparisons.id, { onDelete: "cascade" }),
  modelId: text("model_id").notNull(), modelName: text("model_name").notNull(), stage: text("stage").notNull(),
  content: text("content").notNull().default(""), error: text("error"), latencyMs: integer("latency_ms").notNull().default(0),
  inputTokens: integer("input_tokens"), outputTokens: integer("output_tokens"), metrics: text("metrics").notNull().default("{}"),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});
