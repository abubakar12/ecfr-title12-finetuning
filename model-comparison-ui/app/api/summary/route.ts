import { eq } from "drizzle-orm";
import { getDb } from "@/db";
import { comparisons, responses } from "@/db/schema";

export async function GET(request:Request){
  try{
    const uid=request.headers.get("oai-authenticated-user-id")??"local-user";
    const rows=await getDb().select({modelId:responses.modelId,modelName:responses.modelName,stage:responses.stage,latencyMs:responses.latencyMs,error:responses.error,metrics:responses.metrics,winnerModelId:comparisons.winnerModelId,comparisonId:comparisons.id}).from(responses).innerJoin(comparisons,eq(responses.comparisonId,comparisons.id)).where(eq(comparisons.userId,uid));
    const buckets=new Map<string,{modelId:string;modelName:string;stage:string;runs:number;successful:number;citationScored:number;citationCorrect:number;f1Sum:number;f1Scored:number;latencySum:number;wins:number}>();
    for(const row of rows){const current=buckets.get(row.modelId)??{modelId:row.modelId,modelName:row.modelName,stage:row.stage,runs:0,successful:0,citationScored:0,citationCorrect:0,f1Sum:0,f1Scored:0,latencySum:0,wins:0};const metrics=JSON.parse(row.metrics) as{citationCorrect:boolean|null;referenceF1:number|null};current.runs++;if(!row.error){current.successful++;if(metrics.citationCorrect!==null){current.citationScored++;if(metrics.citationCorrect)current.citationCorrect++}if(metrics.referenceF1!==null){current.f1Scored++;current.f1Sum+=metrics.referenceF1}}current.latencySum+=row.latencyMs;if(row.winnerModelId===row.modelId)current.wins++;buckets.set(row.modelId,current)}
    const models=[...buckets.values()].map((item)=>({...item,citationAccuracy:item.citationScored?item.citationCorrect/item.citationScored:null,referenceF1:item.f1Scored?item.f1Sum/item.f1Scored:null,avgLatencyMs:item.runs?item.latencySum/item.runs:0})).sort((a,b)=>b.wins-a.wins||(b.referenceF1??-1)-(a.referenceF1??-1));
    return Response.json({totalComparisons:new Set(rows.map((row)=>row.comparisonId)).size,models});
  }catch(error){return Response.json({error:error instanceof Error?error.message:"Summary unavailable"},{status:500})}
}
