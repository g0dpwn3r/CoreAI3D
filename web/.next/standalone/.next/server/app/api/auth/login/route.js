"use strict";(()=>{var e={};e.id=873,e.ids=[873],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},3239:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>N,patchFetch:()=>R,requestAsyncStorage:()=>c,routeModule:()=>d,serverHooks:()=>p,staticGenerationAsyncStorage:()=>l});var s={};t.r(s),t.d(s,{POST:()=>u});var a=t(9303),E=t(8716),T=t(670),i=t(7070),n=t(2901),o=t(9178);async function u(e){try{let{email:r,password:t}=await e.json();if(!r||!t)return i.NextResponse.json({error:"Email and password are required"},{status:400});let s=n.Z.prepare("SELECT * FROM users WHERE email = ?").get(r);if(!s||!(0,o.Gv)(t,s.password_hash))return i.NextResponse.json({error:"Invalid credentials"},{status:401});let a=(0,o.RA)({id:s.id,email:s.email,role:s.role}),E=i.NextResponse.json({user:{id:s.id,email:s.email,role:s.role,name:s.name,avatar_url:s.avatar_url},token:a});return E.cookies.set("auth_token",a,{httpOnly:!0,secure:!0,sameSite:"lax",maxAge:604800,path:"/"}),E}catch(e){return console.error("Login error:",e),i.NextResponse.json({error:"Login failed"},{status:500})}}let d=new a.AppRouteRouteModule({definition:{kind:E.x.APP_ROUTE,page:"/api/auth/login/route",pathname:"/api/auth/login",filename:"route",bundlePath:"app/api/auth/login/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/auth/login/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:c,staticGenerationAsyncStorage:l,serverHooks:p}=d,N="/api/auth/login/route";function R(){return(0,T.patchFetch)({serverHooks:p,staticGenerationAsyncStorage:l})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>o,RA:()=>u,WX:()=>d,c_:()=>n,so:()=>c});var s=t(2023),a=t.n(s),E=t(1482),T=t.n(E);let i=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function n(e){return a().hashSync(e,12)}function o(e,r){return a().compareSync(e,r)}function u(e){return T().sign(e,i,{expiresIn:"7d"})}function d(e){try{return T().verify(e,i)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let s=require("better-sqlite3");var a=t.n(s),E=t(1017);let T=require("fs"),i=(0,E.join)(process.cwd(),"data");(0,T.existsSync)(i)||(0,T.mkdirSync)(i,{recursive:!0});let n=(0,E.join)(i,"coreai3d.db"),o=new(a())(n);o.exec(`
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
`);let u=o}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[948,972,944],()=>t(3239));module.exports=s})();