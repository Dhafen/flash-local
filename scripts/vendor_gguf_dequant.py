"""Extract only CPU dequantizers; arithmetic is unchanged from pinned llama.cpp."""
import urllib.request,re,pathlib,hashlib,json
root=pathlib.Path(__file__).resolve().parents[1]
rev='e467e5b5e5ba43071d6151b3d881deb9300d000e'
base=f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{rev}/'
files={p:urllib.request.urlopen(base+p).read() for p in ['ggml/src/ggml-quants.c','ggml/src/ggml-common.h','LICENSE']}
s=files['ggml/src/ggml-quants.c'].decode();parts=[]
for name in ['get_scale_min_k4','dequantize_row_q4_0','dequantize_row_q5_0','dequantize_row_q8_0','dequantize_row_q4_K','dequantize_row_q6_K']:
 match=re.search(r'(?m)^(?:static inline )?void '+name+r'\(',s)
 if not match:raise ValueError(name)
 start=match.start();b=s.index('{',start);depth=1;end=b+1
 while depth:
  if s[end]=='{':depth+=1
  elif s[end]=='}':depth-=1
  end+=1
 text=s[start:end]
 if not text.startswith('static'):text='static '+text
 parts.append(text)
preamble='// Derived verbatim arithmetic from llama.cpp '+rev+' ggml/src/ggml-quants.c\n// Only visibility changed to static; see LICENSE.llama.cpp and provenance.json.\n'
(root/'Sources/CGGUFQuants/vendor/dequant.inc').write_text(preamble+'\n\n'.join(parts)+'\n')
(root/'Sources/CGGUFQuants/vendor/ggml-common.h').write_bytes(files['ggml/src/ggml-common.h'])
(root/'Sources/CGGUFQuants/LICENSE.llama.cpp').write_bytes(files['LICENSE'])
(root/'Sources/CGGUFQuants/provenance.json').write_text(json.dumps({'revision':rev,'source_sha256':{p:hashlib.sha256(b).hexdigest() for p,b in files.items()},'transform':'Extract six named functions and give dequantizers static linkage; no arithmetic edits'},indent=2)+'\n')
