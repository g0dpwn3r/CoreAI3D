"use strict";(()=>{var e={};e.id=628,e.ids=[628],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4300:e=>{e.exports=require("buffer")},6113:e=>{e.exports=require("crypto")},1017:e=>{e.exports=require("path")},2781:e=>{e.exports=require("stream")},3837:e=>{e.exports=require("util")},5105:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>p,patchFetch:()=>I,requestAsyncStorage:()=>_,routeModule:()=>c,serverHooks:()=>N,staticGenerationAsyncStorage:()=>R});var s={};t.r(s),t.d(s,{GET:()=>u,dynamic:()=>d});var E=t(9303),a=t(8716),T=t(670),n=t(7070),i=t(9178),o=t(2901);async function u(e){try{let r=function(e){let r=e.headers.get("authorization"),t=e.cookies.get("auth_token")?.value,s=(0,i.so)(r)||t;return s?(0,i.WX)(s):null}(e);if(!r||"admin"!==r.role)return n.NextResponse.json({error:"Admin access required"},{status:403});let t=o.Z.prepare(`
      SELECT id, email, role, name, avatar_url, created_at,
        (SELECT COUNT(*) FROM conversations WHERE user_id = users.id) as conversations_count,
        (SELECT COUNT(*) FROM beta_keys WHERE used_by = users.id) as keys_used
      FROM users
      ORDER BY created_at DESC
    `).all();return n.NextResponse.json({users:t})}catch(e){return console.error("Get users error:",e),n.NextResponse.json({error:"Failed to fetch users"},{status:500})}}let d="force-dynamic",c=new E.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/admin/users/route",pathname:"/api/admin/users",filename:"route",bundlePath:"app/api/admin/users/route"},resolvedPagePath:"/mnt/data_ssd/CoreAI3D/web/app/api/admin/users/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:_,staticGenerationAsyncStorage:R,serverHooks:N}=c,p="/api/admin/users/route";function I(){return(0,T.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:R})}},9178:(e,r,t)=>{t.d(r,{Gv:()=>o,RA:()=>u,WX:()=>d,c_:()=>i,so:()=>c});var s=t(2023),E=t.n(s),a=t(1482),T=t.n(a);let n=process.env.JWT_SECRET||"coreai3d-super-secret-change-in-production";function i(e){return E().hashSync(e,12)}function o(e,r){return E().compareSync(e,r)}function u(e){return T().sign(e,n,{expiresIn:"7d"})}function d(e){try{return T().verify(e,n)}catch{return null}}function c(e){return e&&e.startsWith("Bearer ")?e.substring(7):null}},2901:(e,r,t)=>{t.d(r,{Z:()=>u});let s=require("better-sqlite3");var E=t.n(s),a=t(1017);let T=require("fs"),n=(0,a.join)(process.cwd(),"data");(0,T.existsSync)(n)||(0,T.mkdirSync)(n,{recursive:!0});let i=(0,a.join)(n,"coreai3d.db"),o=new(E())(i);o.exec(`
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
`);let u=o}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[948,972,944],()=>t(5105));module.exports=s})();