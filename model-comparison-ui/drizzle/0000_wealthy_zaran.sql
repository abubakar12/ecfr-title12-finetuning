CREATE TABLE `comparisons` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`title` text NOT NULL,
	`prompt` text NOT NULL,
	`system_prompt` text NOT NULL,
	`reference_answer` text,
	`expected_citation` text,
	`model_configs` text NOT NULL,
	`temperature_milli` integer DEFAULT 0 NOT NULL,
	`max_tokens` integer DEFAULT 256 NOT NULL,
	`winner_model_id` text,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE `responses` (
	`id` text PRIMARY KEY NOT NULL,
	`comparison_id` text NOT NULL,
	`model_id` text NOT NULL,
	`model_name` text NOT NULL,
	`stage` text NOT NULL,
	`content` text DEFAULT '' NOT NULL,
	`error` text,
	`latency_ms` integer DEFAULT 0 NOT NULL,
	`input_tokens` integer,
	`output_tokens` integer,
	`metrics` text DEFAULT '{}' NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	FOREIGN KEY (`comparison_id`) REFERENCES `comparisons`(`id`) ON UPDATE no action ON DELETE cascade
);
