import { env } from "cloudflare:workers";
import type { ModelConfig } from "@/lib/model-catalog";

type StartRequest={prompt?:string;systemPrompt?:string;temperature?:number;maxTokens?:number;models?:ModelConfig[]};

export async function POST(request:Request){
  try{
    const payload=await request.json() as StartRequest,prompt=payload.prompt?.trim()??"",systemPrompt=payload.systemPrompt?.trim()??"",models=payload.models?.filter((model)=>model.endpointUrl?.startsWith("local://"))??[];
    if(!env.LOCAL_INFERENCE_URL||!env.REGBENCH_GATEWAY_TOKEN)throw new Error("The CPU inference gateway is not configured.");
    if(!prompt||!systemPrompt)return Response.json({error:"Prompt and system prompt are required."},{status:400});
    if(models.length<2||models.length>5)return Response.json({error:"Select between two and five CPU models."},{status:400});
    const temperature=Math.max(0,Math.min(2,Number(payload.temperature??0))),maxTokens=Math.max(8,Math.min(1024,Number(payload.maxTokens??96))),base=env.LOCAL_INFERENCE_URL.replace(/\/$/,"");
    const jobs=await Promise.all(models.map(async(model)=>{const alias=model.endpointUrl!.slice("local://".length)||model.stage,response=await fetch(`${base}/v1/jobs`,{method:"POST",headers:{Authorization:`Bearer ${env.REGBENCH_GATEWAY_TOKEN}`,"Content-Type":"application/json"},body:JSON.stringify({model:alias,messages:[{role:"system",content:systemPrompt},{role:"user",content:prompt}],temperature,max_tokens:maxTokens})}),data=await response.json() as{id?:string;error?:{message?:string}};if(!response.ok||!data.id)throw new Error(data.error?.message||`Could not start ${model.name}`);return{model,jobId:data.id}}));
    return Response.json({jobs},{status:202});
  }catch(error){return Response.json({error:error instanceof Error?error.message:"Could not start comparison"},{status:500})}
}
