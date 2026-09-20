import { desc, eq } from "drizzle-orm";
import { getDb } from "@/db";
import { comparisons } from "@/db/schema";
export async function GET(request:Request){try{const uid=request.headers.get("oai-authenticated-user-id")??"local-user";const rows=await getDb().select({id:comparisons.id,title:comparisons.title,prompt:comparisons.prompt,createdAt:comparisons.createdAt,winnerModelId:comparisons.winnerModelId}).from(comparisons).where(eq(comparisons.userId,uid)).orderBy(desc(comparisons.createdAt)).limit(50);return Response.json({comparisons:rows})}catch(error){return Response.json({error:error instanceof Error?error.message:"History unavailable"},{status:500})}}
