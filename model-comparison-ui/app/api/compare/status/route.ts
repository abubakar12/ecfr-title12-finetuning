import { env } from "cloudflare:workers";
import { getDb } from "@/db";
import { comparisons, responses } from "@/db/schema";
import { evaluateResponse } from "@/lib/evaluation";
import type { ModelConfig } from "@/lib/model-catalog";

type Job={model:ModelConfig;jobId:string};
type CompareRequest={prompt:string;systemPrompt:string;referenceAnswer?:string;expectedCitation?:string;temperature?:number;maxTokens?:number;models:ModelConfig[]};
type JobState={status:"queued"|"running"|"completed"|"failed";result?:{content:string;prompt_tokens:number;completion_tokens:number};error?:string;latency_seconds?:number};

export async function POST(request:Request){
  try{
    const payload=await request.json() as{jobs?:Job[];comparisonRequest?:CompareRequest},jobs=payload.jobs??[],input=payload.comparisonRequest;
    if(!input||jobs.length<2)return Response.json({error:"Invalid comparison job."},{status:400});
    if(!env.LOCAL_INFERENCE_URL||!env.REGBENCH_GATEWAY_TOKEN)throw new Error("The CPU inference gateway is not configured.");
    const base=env.LOCAL_INFERENCE_URL.replace(/\/$/,""),states=await Promise.all(jobs.map(async(job)=>{const response=await fetch(`${base}/v1/jobs/${encodeURIComponent(job.jobId)}`,{headers:{Authorization:`Bearer ${env.REGBENCH_GATEWAY_TOKEN}`}}),data=await response.json() as JobState&{error?:string|{message?:string}};if(!response.ok)throw new Error(typeof data.error==="string"?data.error:data.error?.message||`Could not read ${job.model.name}`);return{job,state:data as JobState}}));
    if(states.some(({state})=>state.status==="queued"||state.status==="running"))return Response.json({status:"running",models:states.map(({job,state})=>({modelId:job.model.id,status:state.status}))},{status:202});
    const rows=states.map(({job,state})=>{const content=state.result?.content?.trim()??"",error=state.status==="failed"?(state.error||"Model job failed"):null;return{model:job.model,content,error,latencyMs:Math.round((state.latency_seconds??0)*1000),inputTokens:state.result?.prompt_tokens??null,outputTokens:state.result?.completion_tokens??null,metrics:evaluateResponse(content,input.referenceAnswer,input.expectedCitation)}}),id=crypto.randomUUID(),temperature=Math.max(0,Math.min(2,Number(input.temperature??0))),maxTokens=Math.max(8,Math.min(1024,Number(input.maxTokens??96))),db=getDb(),uid=request.headers.get("oai-authenticated-user-id")??"local-user";
    await db.insert(comparisons).values({id,userId:uid,title:input.prompt.slice(0,64),prompt:input.prompt,systemPrompt:input.systemPrompt,referenceAnswer:input.referenceAnswer?.trim()||null,expectedCitation:input.expectedCitation?.trim()||null,modelConfigs:JSON.stringify(input.models),temperature:Math.round(temperature*1000),maxTokens});
    await db.insert(responses).values(rows.map((result)=>({id:crypto.randomUUID(),comparisonId:id,modelId:result.model.id,modelName:result.model.name,stage:result.model.stage,content:result.content,error:result.error,latencyMs:result.latencyMs,inputTokens:result.inputTokens,outputTokens:result.outputTokens,metrics:JSON.stringify(result.metrics)})));
    return Response.json({comparison:{id,prompt:input.prompt,systemPrompt:input.systemPrompt,referenceAnswer:input.referenceAnswer??"",expectedCitation:input.expectedCitation??"",temperature,maxTokens,createdAt:new Date().toISOString()},responses:rows});
  }catch(error){return Response.json({error:error instanceof Error?error.message:"Comparison status unavailable"},{status:500})}
}
