"use strict";(()=>{var e={};e.id=297,e.ids=[297],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},9348:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>l,patchFetch:()=>I,requestAsyncStorage:()=>p,routeModule:()=>R,serverHooks:()=>_,staticGenerationAsyncStorage:()=>N});var s={};t.r(s),t.d(s,{DELETE:()=>c,GET:()=>d});var n=t(9303),o=t(8716),E=t(670),a=t(7070),i=t(9178),T=t(2901);function u(e){let r=e.headers.get("authorization"),t=e.cookies.get("auth_token")?.value,s=(0,i.so)(r)||t;return s?(0,i.WX)(s):null}async function d(e,{params:r}){try{let t=u(e);if(!t)return a.NextResponse.json({error:"Unauthorized"},{status:401});let{id:s}=await r,n=T.Z.prepare(`
      SELECT * FROM conversations WHERE id = ? AND user_id = ?
    `).get(s,t.id);if(!n)return a.NextResponse.json({error:"Conversation not found"},{status:404});let o=T.Z.prepare(`
      SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC
    `).all(s);return a.NextResponse.json({conversation:n,messages:o})}catch(e){return console.error("Get conversation error:",e),a.NextResponse.json({error:"Failed to fetch conversation"},{status:500})}}async function c(e,{params:r}){try{let t=u(e);if(!t)return a.NextResponse.json({error:"Unauthorized"},{status:401});let{id:s}=await r,n=T.Z.prepare(`
      DELETE FROM conversations WHERE id = ? AND user_id = ?
    `).run(s,t.id);if(0===n.changes)return a.NextResponse.json({error:"Conversation not found"},{status:404});return a.NextResponse.json({success:!0})}catch(e){return console.error("Delete conversation error:",e),a.NextResponse.json({error:"Failed to delete conversation"},{status:500})}}let R=new n.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/conversations/[id]/route",pathname:"/api/conversations/[id]",filename:"route",bundlePath:"app/api/conversations/[id]/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/conversations/[id]/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:p,staticGenerationAsyncStorage:N,serverHooks:_}=R,l="/api/conversations/[id]/route";function I(){return(0,E.patchFetch)({serverHooks:_,staticGenerationAsyncStorage:N})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>T,RA:()=>u,WX:()=>d,c_:()=>i,so:()=>c});var s=t(2023),n=t.n(s),o=t(1482),E=t.n(o);let a=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function i(e){return n().hashSync(e,12)}function T(e,r){return n().compareSync(e,r)}function u(e){return E().sign(e,a,{expiresIn:"7d"})}function d(e){try{return E().verify(e,a)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let s=require("better-sqlite3");var n=t.n(s),o=t(1017);let E=require("fs"),a=(0,o.join)(process.cwd(),"data");(0,E.existsSync)(a)||(0,E.mkdirSync)(a,{recursive:!0});let i=(0,o.join)(a,"coreai3d.db"),T=new(n())(i);T.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    avatar_url TEXT,
    role TEXT DEFAULT 'user',
    api_key TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS beta_keys (
    id TEXT PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    created_by TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    used_by TEXT,
    used_at DATETIME,
    is_active INTEGER DEFAULT 1,
    notes TEXT,
    FOREIGN KEY (created_by) REFERENCES users(id),
    FOREIGN KEY (used_by) REFERENCES users(id)
  );

  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    model TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
  );

  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
  );

  CREATE INDEX IF NOT EXISTS idx_beta_keys_key ON beta_keys(key);
  CREATE INDEX IF NOT EXISTS idx_beta_keys_used_by ON beta_keys(used_by);
  CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
  CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
`);let u=T}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[948,972,944],()=>t(9348));module.exports=s})();