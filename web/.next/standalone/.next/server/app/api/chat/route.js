"use strict";(()=>{var e={};e.id=744,e.ids=[744],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},3545:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>I,patchFetch:()=>R,requestAsyncStorage:()=>p,routeModule:()=>c,serverHooks:()=>N,staticGenerationAsyncStorage:()=>l});var n={};r.r(n),r.d(n,{POST:()=>u});var a=r(9303),o=r(8716),s=r(670),i=r(7070),E=r(9178),T=r(2901),d=r(9576);async function u(e){try{let t=function(e){let t=e.headers.get("authorization"),r=e.cookies.get("auth_token")?.value,n=(0,E.so)(t)||r;return n?(0,E.WX)(n):null}(e);if(!t)return i.NextResponse.json({error:"Unauthorized"},{status:401});let{message:r,model:n,conversation_id:a,files:o}=await e.json();if(!r||!r.trim())return i.NextResponse.json({error:"Message is required"},{status:400});let s=a;if(!s){s=(0,d.Z)();let e=r.substring(0,50)+(r.length>50?"...":"");T.Z.prepare(`
        INSERT INTO conversations (id, user_id, title, model)
        VALUES (?, ?, ?, ?)
      `).run(s,t.id,e,n||"coreai3d")}T.Z.prepare(`
      INSERT INTO messages (id, conversation_id, role, content)
      VALUES (?, ?, 'user', ?)
    `).run((0,d.Z)(),s,r);let u=new TextEncoder,c=new ReadableStream({async start(e){try{let a=process.env.COREAI3D_API_URL||"http://coreai3d:8080",o=await fetch(`${a}/api/chat`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${t.id}`},body:JSON.stringify({message:r,model:n||"coreai3d",conversation_id:s,stream:!0})});if(!o.ok)throw Error(`Backend error: ${o.status}`);let i=o.body?.getReader();if(!i)throw Error("No response body");let E="",c=!0;for(e.enqueue(u.encode(`data: ${JSON.stringify({type:"thinking",conversation_id:s})}

`));;){let{done:t,value:r}=await i.read();if(t)break;for(let t of new TextDecoder().decode(r).split("\n").filter(e=>e.trim()))if(t.startsWith("data: ")){let r=t.substring(6);try{let t=JSON.parse(r);if("thinking"===t.type&&c)continue;"token"===t.type||t.content?(c&&(c=!1,e.enqueue(u.encode(`data: ${JSON.stringify({type:"generating"})}

`))),E+=t.content||t.token||"",e.enqueue(u.encode(`data: ${JSON.stringify({type:"token",content:t.content||t.token})}

`))):"done"===t.type&&e.enqueue(u.encode(`data: ${JSON.stringify({type:"done",conversation_id:s})}

`))}catch{c&&(c=!1,e.enqueue(u.encode(`data: ${JSON.stringify({type:"generating"})}

`))),E+=r,e.enqueue(u.encode(`data: ${JSON.stringify({type:"token",content:r})}

`))}}}E&&(T.Z.prepare(`
              INSERT INTO messages (id, conversation_id, role, content)
              VALUES (?, ?, 'assistant', ?)
            `).run((0,d.Z)(),s,E),T.Z.prepare(`
              UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
            `).run(s))}catch(t){console.error("Chat stream error:",t),e.enqueue(u.encode(`data: ${JSON.stringify({type:"error",error:"Failed to get response from AI"})}

`))}finally{e.close()}}});return new Response(c,{headers:{"Content-Type":"text/event-stream","Cache-Control":"no-cache",Connection:"keep-alive"}})}catch(e){return console.error("Chat error:",e),i.NextResponse.json({error:"Chat failed"},{status:500})}}let c=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/chat/route",pathname:"/api/chat",filename:"route",bundlePath:"app/api/chat/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/chat/route.ts",nextConfigOutput:"standalone",userland:n}),{requestAsyncStorage:p,staticGenerationAsyncStorage:l,serverHooks:N}=c,I="/api/chat/route";function R(){return(0,s.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:l})}},9178:(e,t,r)=>{r.d(t,{Gv:()=>T,RA:()=>d,WX:()=>u,c_:()=>E,so:()=>c});var n=r(2023),a=r.n(n),o=r(1482),s=r.n(o);let i=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function E(e){return a().hashSync(e,12)}function T(e,t){return a().compareSync(e,t)}function d(e){return s().sign(e,i,{expiresIn:"7d"})}function u(e){try{return s().verify(e,i)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,t,r)=>{r.d(t,{Z:()=>d});let n=require("better-sqlite3");var a=r.n(n),o=r(1017);let s=require("fs"),i=(0,o.join)(process.cwd(),"data");(0,s.existsSync)(i)||(0,s.mkdirSync)(i,{recursive:!0});let E=(0,o.join)(i,"coreai3d.db"),T=new(a())(E);T.exec(`
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
`);let d=T},9576:(e,t,r)=>{r.d(t,{Z:()=>T});var n=r(6113),a=r.n(n);let o={randomUUID:a().randomUUID},s=new Uint8Array(256),i=s.length,E=[];for(let e=0;e<256;++e)E.push((e+256).toString(16).slice(1));let T=function(e,t,r){if(o.randomUUID&&!t&&!e)return o.randomUUID();let n=(e=e||{}).random||(e.rng||function(){return i>s.length-16&&(a().randomFillSync(s),i=0),s.slice(i,i+=16)})();if(n[6]=15&n[6]|64,n[8]=63&n[8]|128,t){r=r||0;for(let e=0;e<16;++e)t[r+e]=n[e];return t}return function(e,t=0){return E[e[t+0]]+E[e[t+1]]+E[e[t+2]]+E[e[t+3]]+"-"+E[e[t+4]]+E[e[t+5]]+"-"+E[e[t+6]]+E[e[t+7]]+"-"+E[e[t+8]]+E[e[t+9]]+"-"+E[e[t+10]]+E[e[t+11]]+E[e[t+12]]+E[e[t+13]]+E[e[t+14]]+E[e[t+15]]}(n)}}};var t=require("../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),n=t.X(0,[948,972,944],()=>r(3545));module.exports=n})();