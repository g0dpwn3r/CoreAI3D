"use strict";(()=>{var e={};e.id=548,e.ids=[548],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},103:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>l,patchFetch:()=>_,requestAsyncStorage:()=>N,routeModule:()=>c,serverHooks:()=>R,staticGenerationAsyncStorage:()=>p});var a={};r.r(a),r.d(a,{GET:()=>d,dynamic:()=>u});var s=r(9303),E=r(8716),T=r(670),i=r(7070),n=r(9178),o=r(2901);async function d(e){try{let t=e.headers.get("authorization"),r=e.cookies.get("auth_token")?.value,a=(0,n.so)(t)||r;if(!a)return i.NextResponse.json({valid:!1,error:"No token provided"},{status:401});let s=(0,n.WX)(a);if(!s)return i.NextResponse.json({valid:!1,error:"Invalid token"},{status:401});let E=o.Z.prepare("SELECT id, email, role, name, avatar_url, api_key FROM users WHERE id = ?").get(s.id);if(!E)return i.NextResponse.json({valid:!1,error:"User not found"},{status:401});return i.NextResponse.json({valid:!0,user:E})}catch(e){return console.error("Validation error:",e),i.NextResponse.json({valid:!1,error:"Validation failed"},{status:500})}}let u="force-dynamic",c=new s.AppRouteRouteModule({definition:{kind:E.x.APP_ROUTE,page:"/api/auth/validate/route",pathname:"/api/auth/validate",filename:"route",bundlePath:"app/api/auth/validate/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/auth/validate/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:N,staticGenerationAsyncStorage:p,serverHooks:R}=c,l="/api/auth/validate/route";function _(){return(0,T.patchFetch)({serverHooks:R,staticGenerationAsyncStorage:p})}},9178:(e,t,r)=>{r.d(t,{Gv:()=>o,RA:()=>d,WX:()=>u,c_:()=>n,so:()=>c});var a=r(2023),s=r.n(a),E=r(1482),T=r.n(E);let i=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function n(e){return s().hashSync(e,12)}function o(e,t){return s().compareSync(e,t)}function d(e){return T().sign(e,i,{expiresIn:"7d"})}function u(e){try{return T().verify(e,i)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,t,r)=>{r.d(t,{Z:()=>d});let a=require("better-sqlite3");var s=r.n(a),E=r(1017);let T=require("fs"),i=(0,E.join)(process.cwd(),"data");(0,T.existsSync)(i)||(0,T.mkdirSync)(i,{recursive:!0});let n=(0,E.join)(i,"coreai3d.db"),o=new(s())(n);o.exec(`
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
`);let d=o}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),a=t.X(0,[948,972,944],()=>r(103));module.exports=a})();