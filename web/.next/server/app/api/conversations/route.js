"use strict";(()=>{var e={};e.id=3,e.ids=[3],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},2430:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>I,patchFetch:()=>A,requestAsyncStorage:()=>N,routeModule:()=>p,serverHooks:()=>_,staticGenerationAsyncStorage:()=>l});var s={};t.r(s),t.d(s,{GET:()=>c,POST:()=>R});var n=t(9303),o=t(8716),a=t(670),E=t(7070),i=t(9178),T=t(2901),u=t(9576);function d(e){let r=e.headers.get("authorization"),t=e.cookies.get("auth_token")?.value,s=(0,i.so)(r)||t;return s?(0,i.WX)(s):null}async function c(e){try{let r=d(e);if(!r)return E.NextResponse.json({error:"Unauthorized"},{status:401});let t=T.Z.prepare(`
      SELECT c.*, 
        (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message,
        (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
      FROM conversations c
      WHERE c.user_id = ?
      ORDER BY c.updated_at DESC
    `).all(r.id);return E.NextResponse.json({conversations:t})}catch(e){return console.error("Get conversations error:",e),E.NextResponse.json({error:"Failed to fetch conversations"},{status:500})}}async function R(e){try{let r=d(e);if(!r)return E.NextResponse.json({error:"Unauthorized"},{status:401});let{title:t,model:s}=await e.json(),n=(0,u.Z)();T.Z.prepare(`
      INSERT INTO conversations (id, user_id, title, model)
      VALUES (?, ?, ?, ?)
    `).run(n,r.id,t||"New Chat",s||"coreai3d");let o=T.Z.prepare("SELECT * FROM conversations WHERE id = ?").get(n);return E.NextResponse.json({conversation:o})}catch(e){return console.error("Create conversation error:",e),E.NextResponse.json({error:"Failed to create conversation"},{status:500})}}let p=new n.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/conversations/route",pathname:"/api/conversations",filename:"route",bundlePath:"app/api/conversations/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/conversations/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:N,staticGenerationAsyncStorage:l,serverHooks:_}=p,I="/api/conversations/route";function A(){return(0,a.patchFetch)({serverHooks:_,staticGenerationAsyncStorage:l})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>T,RA:()=>u,WX:()=>d,c_:()=>i,so:()=>c});var s=t(2023),n=t.n(s),o=t(1482),a=t.n(o);let E=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function i(e){return n().hashSync(e,12)}function T(e,r){return n().compareSync(e,r)}function u(e){return a().sign(e,E,{expiresIn:"7d"})}function d(e){try{return a().verify(e,E)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let s=require("better-sqlite3");var n=t.n(s),o=t(1017);let a=require("fs"),E=(0,o.join)(process.cwd(),"data");(0,a.existsSync)(E)||(0,a.mkdirSync)(E,{recursive:!0});let i=(0,o.join)(E,"coreai3d.db"),T=new(n())(i);T.exec(`
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
`);let u=T},9576:(e,r,t)=>{t.d(r,{Z:()=>T});var s=t(6113),n=t.n(s);let o={randomUUID:n().randomUUID},a=new Uint8Array(256),E=a.length,i=[];for(let e=0;e<256;++e)i.push((e+256).toString(16).slice(1));let T=function(e,r,t){if(o.randomUUID&&!r&&!e)return o.randomUUID();let s=(e=e||{}).random||(e.rng||function(){return E>a.length-16&&(n().randomFillSync(a),E=0),a.slice(E,E+=16)})();if(s[6]=15&s[6]|64,s[8]=63&s[8]|128,r){t=t||0;for(let e=0;e<16;++e)r[t+e]=s[e];return r}return function(e,r=0){return i[e[r+0]]+i[e[r+1]]+i[e[r+2]]+i[e[r+3]]+"-"+i[e[r+4]]+i[e[r+5]]+"-"+i[e[r+6]]+i[e[r+7]]+"-"+i[e[r+8]]+i[e[r+9]]+"-"+i[e[r+10]]+i[e[r+11]]+i[e[r+12]]+i[e[r+13]]+i[e[r+14]]+i[e[r+15]]}(s)}}};var r=require("../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[948,972,944],()=>t(2430));module.exports=s})();