"use strict";(()=>{var e={};e.id=70,e.ids=[70],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},9132:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>I,patchFetch:()=>A,requestAsyncStorage:()=>_,routeModule:()=>R,serverHooks:()=>l,staticGenerationAsyncStorage:()=>N});var s={};t.r(s),t.d(s,{GET:()=>d,PUT:()=>c,dynamic:()=>p});var E=t(9303),a=t(8716),T=t(670),i=t(7070),n=t(9178),o=t(2901);function u(e){let r=e.headers.get("authorization"),t=e.cookies.get("auth_token")?.value,s=(0,n.so)(r)||t;return s?(0,n.WX)(s):null}async function d(e){try{let r=u(e);if(!r)return i.NextResponse.json({error:"Unauthorized"},{status:401});let t=o.Z.prepare(`
      SELECT id, email, role, name, avatar_url, api_key, created_at
      FROM users WHERE id = ?
    `).get(r.id),s=o.Z.prepare(`
      SELECT 
        (SELECT COUNT(*) FROM conversations WHERE user_id = ?) as conversations_count,
        (SELECT COUNT(*) FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)) as messages_count
    `).get(r.id,r.id);return i.NextResponse.json({user:t,stats:s})}catch(e){return console.error("Get profile error:",e),i.NextResponse.json({error:"Failed to fetch profile"},{status:500})}}async function c(e){try{let r=u(e);if(!r)return i.NextResponse.json({error:"Unauthorized"},{status:401});let{name:t,avatar_url:s}=await e.json();o.Z.prepare(`
      UPDATE users SET name = ?, avatar_url = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
    `).run(t,s,r.id);let E=o.Z.prepare(`
      SELECT id, email, role, name, avatar_url, api_key, created_at
      FROM users WHERE id = ?
    `).get(r.id);return i.NextResponse.json({user:E})}catch(e){return console.error("Update profile error:",e),i.NextResponse.json({error:"Failed to update profile"},{status:500})}}let p="force-dynamic",R=new E.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/user/profile/route",pathname:"/api/user/profile",filename:"route",bundlePath:"app/api/user/profile/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/user/profile/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:_,staticGenerationAsyncStorage:N,serverHooks:l}=R,I="/api/user/profile/route";function A(){return(0,T.patchFetch)({serverHooks:l,staticGenerationAsyncStorage:N})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>o,RA:()=>u,WX:()=>d,c_:()=>n,so:()=>c});var s=t(2023),E=t.n(s),a=t(1482),T=t.n(a);let i=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function n(e){return E().hashSync(e,12)}function o(e,r){return E().compareSync(e,r)}function u(e){return T().sign(e,i,{expiresIn:"7d"})}function d(e){try{return T().verify(e,i)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let s=require("better-sqlite3");var E=t.n(s),a=t(1017);let T=require("fs"),i=(0,a.join)(process.cwd(),"data");(0,T.existsSync)(i)||(0,T.mkdirSync)(i,{recursive:!0});let n=(0,a.join)(i,"coreai3d.db"),o=new(E())(n);o.exec(`
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
`);let u=o}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[948,972,944],()=>t(9132));module.exports=s})();