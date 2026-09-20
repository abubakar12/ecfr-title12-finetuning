declare namespace Cloudflare {
  interface Env {
    DB?: D1Database;
    HF_TOKEN?: string;
    LOCAL_INFERENCE_URL?: string;
    REGBENCH_GATEWAY_TOKEN?: string;
    BUCKET?: R2Bucket;
  }
}
