"use strict";(()=>{var e={};e.id=969,e.ids=[969],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},3497:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>_,patchFetch:()=>R,requestAsyncStorage:()=>p,routeModule:()=>c,serverHooks:()=>N,staticGenerationAsyncStorage:()=>I});var n={};t.r(n),t.d(n,{POST:()=>d});var a=t(9303),s=t(8716),E=t(670),T=t(7070),i=t(9178),o=t(2901),u=t(9576);async function d(e){try{let r=function(e){let r=e.headers.get("authorization"),t=e.cookies.get("auth_token")?.value,n=(0,i.so)(r)||t;return n?(0,i.WX)(n):null}(e);if(!r)return T.NextResponse.json({error:"Unauthorized"},{status:401});let t=`ca3d_${(0,u.Z)().replace(/-/g,"")}`;return o.Z.prepare(`
      UPDATE users SET api_key = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
    `).run(t,r.id),T.NextResponse.json({api_key:t})}catch(e){return console.error("Regenerate API key error:",e),T.NextResponse.json({error:"Failed to regenerate API key"},{status:500})}}let c=new a.AppRouteRouteModule({definition:{kind:s.x.APP_ROUTE,page:"/api/user/regenerate-api-key/route",pathname:"/api/user/regenerate-api-key",filename:"route",bundlePath:"app/api/user/regenerate-api-key/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/user/regenerate-api-key/route.ts",nextConfigOutput:"standalone",userland:n}),{requestAsyncStorage:p,staticGenerationAsyncStorage:I,serverHooks:N}=c,_="/api/user/regenerate-api-key/route";function R(){return(0,E.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:I})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>o,RA:()=>u,WX:()=>d,c_:()=>i,so:()=>c});var n=t(2023),a=t.n(n),s=t(1482),E=t.n(s);let T=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function i(e){return a().hashSync(e,12)}function o(e,r){return a().compareSync(e,r)}function u(e){return E().sign(e,T,{expiresIn:"7d"})}function d(e){try{return E().verify(e,T)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let n=require("better-sqlite3");var a=t.n(n),s=t(1017);let E=require("fs"),T=(0,s.join)(process.cwd(),"data");(0,E.existsSync)(T)||(0,E.mkdirSync)(T,{recursive:!0});let i=(0,s.join)(T,"coreai3d.db"),o=new(a())(i);o.exec(`
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
`);let u=o},9576:(e,r,t)=>{t.d(r,{Z:()=>o});var n=t(6113),a=t.n(n);let s={randomUUID:a().randomUUID},E=new Uint8Array(256),T=E.length,i=[];for(let e=0;e<256;++e)i.push((e+256).toString(16).slice(1));let o=function(e,r,t){if(s.randomUUID&&!r&&!e)return s.randomUUID();let n=(e=e||{}).random||(e.rng||function(){return T>E.length-16&&(a().randomFillSync(E),T=0),E.slice(T,T+=16)})();if(n[6]=15&n[6]|64,n[8]=63&n[8]|128,r){t=t||0;for(let e=0;e<16;++e)r[t+e]=n[e];return r}return function(e,r=0){return i[e[r+0]]+i[e[r+1]]+i[e[r+2]]+i[e[r+3]]+"-"+i[e[r+4]]+i[e[r+5]]+"-"+i[e[r+6]]+i[e[r+7]]+"-"+i[e[r+8]]+i[e[r+9]]+"-"+i[e[r+10]]+i[e[r+11]]+i[e[r+12]]+i[e[r+13]]+i[e[r+14]]+i[e[r+15]]}(n)}}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),n=r.X(0,[948,972,944],()=>t(3497));module.exports=n})();