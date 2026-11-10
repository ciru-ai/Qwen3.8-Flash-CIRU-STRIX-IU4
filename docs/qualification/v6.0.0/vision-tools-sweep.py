#!/usr/bin/env python3
"""Image + N tool definitions sweep. usage: vision_tools.py --port P --img a.jpg:kw1,kw2 ... [--tiers 0,4,8,16,32] [--rounds 1] --out f.json"""
import argparse, base64, http.client, json, os, sys
NAMES = ["bash","glob","grep","read_file","write_file","edit_file","list_dir","web_fetch","web_search","todo_write",
 "task","notebook_edit","git_status","git_diff","git_commit","run_tests","read_image","screenshot","open_url","kill_shell",
 "bash_output","move_file","delete_file","apply_patch","search_symbols","find_references","format_code","lint","http_request","sql_query",
 "memory_read","memory_write"]
DESC = {
 "bash":"Execute a shell command in the workspace and return stdout/stderr. Use for builds, scripts and inspection.",
 "glob":"Find files in the workspace whose paths match a glob pattern such as **/*.py.",
 "grep":"Search file contents with a regular expression and return matching lines.",
 "read_file":"Read a text file from the workspace and return its contents with line numbers.",
 "read_image":"Load an image file from the workspace by path and return it for viewing.",
}
def tool(n):
    props = {"path":{"type":"string","description":"Workspace-relative path."},
             "query":{"type":"string","description":"Command, pattern or text argument."},
             "timeout":{"type":"integer","description":"Seconds before the call is cancelled."},
             "recursive":{"type":"boolean","description":"Whether to descend into subdirectories."}}
    return {"type":"function","function":{"name":n,
        "description":DESC.get(n, f"Tool `{n}` of the coding agent. " + "Performs the operation its name describes on the current workspace and returns a structured result. "*2),
        "parameters":{"type":"object","properties":props,"required":["query"]}}}
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=18181); ap.add_argument("--img", action="append", required=True)
ap.add_argument("--tiers", default="0,4,8,16,32"); ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--prompt", default="这张图片里是什么？"); ap.add_argument("--think", type=int, default=0)
ap.add_argument("--out", default=None); ap.add_argument("--max", type=int, default=200); ap.add_argument("--pad", type=int, default=0); ap.add_argument("--after", type=int, default=0)
a = ap.parse_args()
key = os.environ.get("LLAMA_API_KEY", "")
res = []
for spec in a.img:
    path, _, kws = spec.partition(":"); kws = [k for k in kws.split(",") if k]
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    for n in [int(x) for x in a.tiers.split(",")]:
        for r in range(a.rounds):
            body = {"model":"main","temperature":0,"stream":False,"max_tokens":a.max,
                    "chat_template_kwargs":{"enable_thinking":bool(a.think)},
                    "messages":[{"role":"system","content":"You are a helpful assistant." + (" Notes: " + " ".join(f"Rule {i}: keep answers accurate, concise and grounded in what the user provided." for i in range(a.pad)) if a.pad else "")},
                                {"role":"user","content":[{"type":"text","text":a.prompt},
                                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}]}
            if n: body["tools"] = [tool(x) for x in NAMES[:n]]
            c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type":"application/json","Authorization":"Bearer "+key})
            d = json.loads(c.getresponse().read()); c.close()
            if "choices" not in d:
                print(os.path.basename(path), n, "ERROR", str(d)[:200]); continue
            m = d["choices"][0]["message"]; content = m.get("content") or ""
            calls = [t["function"]["name"] + "(" + t["function"]["arguments"][:60] + ")" for t in (m.get("tool_calls") or [])]
            ok = (not calls) and (any(k.lower() in content.lower() for k in kws) if kws else bool(content))
            pt = d.get("usage", {}).get("prompt_tokens")
            res.append({"img":os.path.basename(path),"tools":n,"round":r,"ok":ok,"calls":calls,"content":content,"pt":pt})
            print(f"{os.path.basename(path):10s} tools={n:2d} pt={pt} {'OK  ' if ok else 'FAIL'} calls={calls} | {content[:90]!r}", flush=True)
tot = {}
for x in res: tot.setdefault(x["tools"], [0,0]); tot[x["tools"]][1]+=1; tot[x["tools"]][0]+=x["ok"]
print("SUMMARY", " ".join(f"t{n}={p}/{t}" for n,(p,t) in sorted(tot.items())))
if a.out: json.dump(res, open(a.out,"w"), ensure_ascii=False, indent=1)
