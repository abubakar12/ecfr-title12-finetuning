import { env } from "cloudflare:workers";
import { getDb } from "@/db";
import { comparisons, responses } from "@/db/schema";
import { evaluateResponse } from "@/lib/evaluation";
import type { ModelConfig } from "@/lib/model-catalog";

type CompareRequest = { prompt?:string;systemPrompt?:string;referenceAnswer?:string;expectedCitation?:string;temperature?:number;maxTokens?:number;models?:ModelConfig[] };
function userId(request:Request){return request.headers.get("oai-authenticated-user-id")??"local-user"}
function endpointFor(model:ModelConfig){
  if(model.endpointUrl?.startsWith("local://")){
    if(!env.LOCAL_INFERENCE_URL)throw new Error("LOCAL_INFERENCE_URL is not configured for this deployment.");
    const alias=model.endpointUrl.slice("local://".length)||model.stage;
    return {url:`${env.LOCAL_INFERENCE_URL.replace(/\/$/,"")}/v1/chat/completions`,requestModel:alias,authorization:env.REGBENCH_GATEWAY_TOKEN||null};
  }
  if(!model.endpointUrl)return {url:"https://router.huggingface.co/v1/chat/completions",requestModel:model.modelId,authorization:env.HF_TOKEN||null};
  const url=new URL(model.endpointUrl),isHuggingFace=url.hostname==="huggingface.co"||url.hostname.endsWith(".huggingface.co")||url.hostname.endsWith(".huggingface.cloud"),isRegBenchTunnel=url.hostname.endsWith(".trycloudflare.com");
  if(url.protocol!=="https:"||(!isHuggingFace&&!isRegBenchTunnel))throw new Error("Endpoint must use HTTPS on a Hugging Face or RegBench tunnel domain.");
  return {url:url.pathname.endsWith("/chat/completions")?url.toString():`${url.toString().replace(/\/$/,"")}/v1/chat/completions`,requestModel:isRegBenchTunnel?model.stage:model.modelId,authorization:isRegBenchTunnel?(env.REGBENCH_GATEWAY_TOKEN||null):(env.HF_TOKEN||null)};
}

async function runModel(model:ModelConfig,body:{prompt:string;systemPrompt:string;temperature:number;maxTokens:number}){
  const started=Date.now();
  try{
    if(!model.modelId.trim()&&!model.endpointUrl)throw new Error("Add a Hugging Face model ID or endpoint URL in Models.");
    const endpoint=endpointFor(model);
    if(!endpoint.authorization&&!model.endpointUrl?.startsWith("local://"))throw new Error("An inference token is not configured for this endpoint.");
    const response=await fetch(endpoint.url,{method:"POST",headers:{...(endpoint.authorization?{Authorization:`Bearer ${endpoint.authorization}`}:{ }),"Content-Type":"application/json"},body:JSON.stringify({model:endpoint.requestModel||undefined,messages:[{role:"system",content:body.systemPrompt},{role:"user",content:body.prompt}],temperature:body.temperature,max_tokens:body.maxTokens,stream:false})});
    const payload=await response.json() as {choices?:{message?:{content?:string}}[];usage?:{prompt_tokens?:number;completion_tokens?:number};error?:string|{message?:string};message?:string};
    const apiError=typeof payload.error==="string"?payload.error:payload.error?.message;
    if(!response.ok)throw new Error(apiError||payload.message||`Hugging Face returned ${response.status}`);
    return {model,content:payload.choices?.[0]?.message?.content?.trim()??"",error:null,latencyMs:Date.now()-started,inputTokens:payload.usage?.prompt_tokens??null,outputTokens:payload.usage?.completion_tokens??null};
  }catch(error){return {model,content:"",error:error instanceof Error?error.message:"Model request failed",latencyMs:Date.now()-started,inputTokens:null,outputTokens:null}}
}

export async function POST(request:Request){
  try{
    const payload=await request.json() as CompareRequest,prompt=payload.prompt?.trim()??"",systemPrompt=payload.systemPrompt?.trim()??"";
    const models=payload.models?.filter((model)=>model.name&&(model.modelId||model.endpointUrl))??[];
    const temperature=Math.max(0,Math.min(2,Number(payload.temperature??0))),maxTokens=Math.max(32,Math.min(1024,Number(payload.maxTokens??256)));
    if(!prompt||!systemPrompt)return Response.json({error:"Prompt and system prompt are required."},{status:400});
    if(models.length<2||models.length>5)return Response.json({error:"Select between two and five configured models."},{status:400});
    const id=crypto.randomUUID(),results=await Promise.all(models.map((model)=>runModel(model,{prompt,systemPrompt,temperature,maxTokens})));
    const responseRows=results.map((result)=>({...result,metrics:evaluateResponse(result.content,payload.referenceAnswer,payload.expectedCitation)}));
    const db=getDb();
    await db.insert(comparisons).values({id,userId:userId(request),title:prompt.slice(0,64),prompt,systemPrompt,referenceAnswer:payload.referenceAnswer?.trim()||null,expectedCitation:payload.expectedCitation?.trim()||null,modelConfigs:JSON.stringify(models),temperature:Math.round(temperature*1000),maxTokens});
    await db.insert(responses).values(responseRows.map((result)=>({id:crypto.randomUUID(),comparisonId:id,modelId:result.model.id,modelName:result.model.name,stage:result.model.stage,content:result.content,error:result.error,latencyMs:result.latencyMs,inputTokens:result.inputTokens,outputTokens:result.outputTokens,metrics:JSON.stringify(result.metrics)})));
    return Response.json({comparison:{id,prompt,systemPrompt,referenceAnswer:payload.referenceAnswer??"",expectedCitation:payload.expectedCitation??"",temperature,maxTokens,createdAt:new Date().toISOString()},responses:responseRows});
  }catch(error){return Response.json({error:error instanceof Error?error.message:"Comparison failed"},{status:500})}
}
