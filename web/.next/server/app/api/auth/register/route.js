"use strict";(()=>{var e={};e.id=2,e.ids=[2],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},5680:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>A,patchFetch:()=>y,requestAsyncStorage:()=>R,routeModule:()=>N,serverHooks:()=>I,staticGenerationAsyncStorage:()=>_});var a={};t.r(a),t.d(a,{POST:()=>p});var s=t(9303),n=t(8716),i=t(670),o=t(7070),E=t(2901),T=t(9178),u=t(9576);let d=process.env.COREAI3D_API_URL||"http://coreai3d:8080";async function c(e){try{let r=await fetch(`${d}/api/v1/beta/validate`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:e})});return await r.json()}catch(e){return console.error("Beta key validation error:",e),{valid:!1,error:"Validation service unavailable"}}}async function l(e,r){try{let t=await fetch(`${d}/api/v1/beta/use`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:e,user_id:r})}),a=await t.json();return!0===a.success}catch(e){return console.error("Beta key usage error:",e),!1}}async function p(e){try{let{email:r,password:t,beta_key:a}=await e.json();if(!r||!t)return o.NextResponse.json({error:"Email and password are required"},{status:400});if(!a)return o.NextResponse.json({error:"Beta key is required for registration"},{status:400});let s=await c(a);if(!s.valid)return o.NextResponse.json({error:s.error},{status:400});if(E.Z.prepare("SELECT id FROM users WHERE email = ?").get(r))return o.NextResponse.json({error:"Email already registered"},{status:400});let n=(0,u.Z)(),i=(0,T.c_)(t),d=`ca3d_${(0,u.Z)().replace(/-/g,"")}`;E.Z.prepare(`
      INSERT INTO users (id, email, password_hash, api_key, role)
      VALUES (?, ?, ?, ?, 'user')
    `).run(n,r,i,d),await l(a,n)||console.warn("Failed to mark beta key as used in backend, but user created");let p=(0,T.RA)({id:n,email:r,role:"user"}),N=o.NextResponse.json({user:{id:n,email:r,role:"user"},token:p});return N.cookies.set("auth_token",p,{httpOnly:!0,secure:!0,sameSite:"lax",maxAge:604800,path:"/"}),N}catch(e){return console.error("Registration error:",e),o.NextResponse.json({error:"Registration failed"},{status:500})}}let N=new s.AppRouteRouteModule({definition:{kind:n.x.APP_ROUTE,page:"/api/auth/register/route",pathname:"/api/auth/register",filename:"route",bundlePath:"app/api/auth/register/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/auth/register/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:R,staticGenerationAsyncStorage:_,serverHooks:I}=N,A="/api/auth/register/route";function y(){return(0,i.patchFetch)({serverHooks:I,staticGenerationAsyncStorage:_})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>T,RA:()=>u,WX:()=>d,c_:()=>E,so:()=>c});var a=t(2023),s=t.n(a),n=t(1482),i=t.n(n);let o=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function E(e){return s().hashSync(e,12)}function T(e,r){return s().compareSync(e,r)}function u(e){return i().sign(e,o,{expiresIn:"7d"})}function d(e){try{return i().verify(e,o)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let a=require("better-sqlite3");var s=t.n(a),n=t(1017);let i=require("fs"),o=(0,n.join)(process.cwd(),"data");(0,i.existsSync)(o)||(0,i.mkdirSync)(o,{recursive:!0});let E=(0,n.join)(o,"coreai3d.db"),T=new(s())(E);T.exec(`
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
`);let u=T},9576:(e,r,t)=>{t.d(r,{Z:()=>T});var a=t(6113),s=t.n(a);let n={randomUUID:s().randomUUID},i=new Uint8Array(256),o=i.length,E=[];for(let e=0;e<256;++e)E.push((e+256).toString(16).slice(1));let T=function(e,r,t){if(n.randomUUID&&!r&&!e)return n.randomUUID();let a=(e=e||{}).random||(e.rng||function(){return o>i.length-16&&(s().randomFillSync(i),o=0),i.slice(o,o+=16)})();if(a[6]=15&a[6]|64,a[8]=63&a[8]|128,r){t=t||0;for(let e=0;e<16;++e)r[t+e]=a[e];return r}return function(e,r=0){return E[e[r+0]]+E[e[r+1]]+E[e[r+2]]+E[e[r+3]]+"-"+E[e[r+4]]+E[e[r+5]]+"-"+E[e[r+6]]+E[e[r+7]]+"-"+E[e[r+8]]+E[e[r+9]]+"-"+E[e[r+10]]+E[e[r+11]]+E[e[r+12]]+E[e[r+13]]+E[e[r+14]]+E[e[r+15]]}(a)}}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),a=r.X(0,[948,972,944],()=>t(5680));module.exports=a})();