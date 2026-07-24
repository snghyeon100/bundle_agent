SELECTED_OPERATORS = ["BuildBundleImageCentroid", "ComputeCandidateMultimodalTextMargins"]
INTENT = "complete an outfit pairing a pleated knit dress with short heeled boots, suggesting visually and textually compatible accessories or garments"
STRATEGY = {"name": "image-centroid-plus-multimodal-margin", "description": "Build an image centroid from the partial bundle's item image embeddings (data/content_feature.pt) and then compute per-candidate image similarity to that centroid plus text-description similarity to the partial-items' description centroid (data/description_feature.pt). Report source-grounded similarity diagnostics from the two files showing candidate-to-bundle relations."}

import json
import os
import math
import torch
from typing import List, Dict

DATA_DIR = "data"
CONTENT_FEAT_PATH = os.path.join(DATA_DIR, "content_feature.pt")
DESCRIPTION_FEAT_PATH = os.path.join(DATA_DIR, "description_feature.pt")
ITEM_INFO_PATH = os.path.join(DATA_DIR, "item_info.json")
OUTPUT_PATH = "output/operator_evidence_bundle9388.json"

PARTIAL_ITEMS = [
{"item_id": 27786, "text": "限时7.5折 所有外套都配 百褶抽绳半高领打底针织连衣裙"},
{"item_id": 4787, "text": "小跟短靴女2018新款网红百搭方头粗跟冬季加绒及裸靴短筒高跟靴子"},
]

CANDIDATES = [
{"label": "A", "item_id": 9877, "text": "于momo2018春装新款港味开叉针织连衣裙外穿长款毛衣裙过膝打底裙"},
{"label": "B", "item_id": 21880, "text": "小包包女2018新款潮韩版夏天夏季百搭迷你小香风菱格链条斜挎小包"},
{"label": "C", "item_id": 26773, "text": "钉珠羽毛图案蕾丝拼接木耳边翻领长袖衬衫"},
{"label": "D", "item_id": 14120, "text": "梦梦家女装冬季2018新款韩版高腰显瘦圆点百褶A字半身裙中长款@"},
{"label": "E", "item_id": 1368, "text": "马衔扣包头半拖鞋女夏外穿2018新款真皮平底无后跟懒人穆勒凉拖鞋"},
{"label": "F", "item_id": 2179, "text": "2018新款夏季长裙子慵懒风chic连衣裙imiss超仙女V领纯蓝色娃娃衫"},
{"label": "G", "item_id": 9204, "text": "满衣大码早秋牛仔裤女2018新款文艺贴布刺绣垮裤复古水洗做旧长裤"},
{"label": "H", "item_id": 30353, "text": "春秋女装外套chic港风pu机车外套女显瘦皮夹克高腰粉色皮衣女短款"},
{"label": "I", "item_id": 2839, "text": "2018春季新款绑带a字裙高腰学生牛仔裙白色半身裙矮小个子短裙女"},
{"label": "J", "item_id": 27985, "text": "ins网红贝雷帽女春秋韩版日系百搭画家帽黑色英伦八角帽女帽子夏"},
]

def load_sources():
    sources = {}
    # load content features (image embeddings)
    if os.path.exists(CONTENT_FEAT_PATH):
        sources["content_feature"] = torch.load(CONTENT_FEAT_PATH, map_location="cpu")
    else:
        sources["content_feature"] = None
    # load description features (text embeddings)
    if os.path.exists(DESCRIPTION_FEAT_PATH):
        sources["description_feature"] = torch.load(DESCRIPTION_FEAT_PATH, map_location="cpu")
    else:
        sources["description_feature"] = None
    # load item metadata titles
    if os.path.exists(ITEM_INFO_PATH):
        with open(ITEM_INFO_PATH, "r", encoding="utf-8") as f:
            sources["item_info"] = json.load(f)
    else:
        sources["item_info"] = {}
    return sources

def safe_index_tensor(tensor, idx):
    # item ids are integer item_id and should index the rows; validate bounds
    if tensor is None:
        return None
    if idx < 0 or idx >= tensor.shape[0]:
        return None
    return tensor[idx]

def l2_normalize(vec):
    if vec is None:
        return None
    norm = float(torch.norm(vec).item())
    if norm == 0:
        return vec
    return vec / norm

def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(torch.dot(a_n, b_n).item())

def build_bundle_image_centroid(partial_item_ids: List[int], content_tensor):
    # Build centroid as elementwise mean of the partial items' image embeddings
    vectors = []
    missing = []
    for iid in partial_item_ids:
        v = safe_index_tensor(content_tensor, iid)
        if v is None:
            missing.append(iid)
        else:
            vectors.append(v.float())
    if len(vectors) == 0:
        return {"centroid": None, "missing": missing, "count": 0}
    stacked = torch.stack(vectors, dim=0)
    centroid = torch.mean(stacked, dim=0)
    return {"centroid": centroid, "missing": missing, "count": len(vectors)}

def compute_candidate_multimodal_text_margins(bundle_image_centroid, partial_desc_centroid, candidate_item_id, sources):
    res = {}
    content_tensor = sources.get("content_feature")
    description_tensor = sources.get("description_feature")
    # image similarity to bundle centroid
    cand_img = safe_index_tensor(content_tensor, candidate_item_id)
    img_sim = None
    if bundle_image_centroid is not None and cand_img is not None:
        img_sim = cosine_similarity(bundle_image_centroid, cand_img.float())
    # text similarity to partial description centroid
    cand_desc = safe_index_tensor(description_tensor, candidate_item_id)
    txt_sim = None
    if partial_desc_centroid is not None and cand_desc is not None:
        txt_sim = cosine_similarity(partial_desc_centroid, cand_desc.float())
    res["image_similarity"] = img_sim
    res["text_similarity"] = txt_sim
    return res

def retrieve_partial_bundle_context(partial_items, sources):
    evidence = []
    partial_ids = [p["item_id"] for p in partial_items]
    content_tensor = sources.get("content_feature")
    description_tensor = sources.get("description_feature")
    # Build image centroid using BuildBundleImageCentroid operator
    bic = build_bundle_image_centroid(partial_ids, content_tensor)
    centroid = bic["centroid"]
    # Build description centroid (mean text embedding) to support multimodal margins step
    desc_vectors = []
    desc_missing = []
    for iid in partial_ids:
        v = safe_index_tensor(description_tensor, iid)
        if v is None:
            desc_missing.append(iid)
        else:
            desc_vectors.append(v.float())
    if len(desc_vectors) > 0:
        desc_centroid = torch.mean(torch.stack(desc_vectors, dim=0), dim=0)
    else:
        desc_centroid = None
    # Prepare evidence strings (max 5)
    # 1) source: concrete partial items found in item_info.json
    try:
        with open(ITEM_INFO_PATH, "r", encoding="utf-8") as f:
            item_info = json.load(f)
    except:
        item_info = {}
    for iid in partial_ids:
        key = str(iid)
        title = item_info.get(key, {}).get("title", "") if item_info else ""
        evidence.append(f"item_info.json: item {iid} title -> '{title}'")
        if len(evidence) >= 3:
            break
    # 2) content_feature centroid diagnostics
    if centroid is not None:
        norm = float(torch.norm(centroid).item())
        evidence.append(f"content_feature.pt: built image centroid from items {partial_ids} -> centroid_norm={norm:.4f}")
    else:
        evidence.append(f"content_feature.pt: image centroid could not be built (missing rows for items {bic.get('missing')})")
    # 3) description centroid diagnostics
    if desc_centroid is not None:
        dn = float(torch.norm(desc_centroid).item())
        evidence.append(f"description_feature.pt: built text centroid from items {partial_ids} -> centroid_norm={dn:.4f}")
    else:
        evidence.append(f"description_feature.pt: text centroid could not be built (missing description rows for some partial items)")
    # Deduplicate and limit to 5
    unique = []
    for e in evidence:
        if e not in unique:
            unique.append(e)
    return {"evidence": unique[:5], "image_centroid": centroid, "text_centroid": desc_centroid, "partial_ids": partial_ids}

def retrieve_candidate_evidence(candidate, partial_items, partial_bundle_context, sources):
    # Use ComputeCandidateMultimodalTextMargins operator
    item_id = candidate["item_id"]
    img_centroid = partial_bundle_context.get("image_centroid")
    txt_centroid = partial_bundle_context.get("text_centroid")
    margins = compute_candidate_multimodal_text_margins(img_centroid, txt_centroid, item_id, sources)
    ev = []
    # Build evidence strings grounded to files
    if margins["image_similarity"] is not None:
        ev.append(f"content_feature.pt: cosine(image_centroid, item {item_id}) = {margins['image_similarity']:.4f} -> visual compatibility metric")
    else:
        ev.append(f"content_feature.pt: missing image embedding for item {item_id} or centroid unavailable -> no visual similarity")
    if margins["text_similarity"] is not None:
        ev.append(f"description_feature.pt: cosine(text_centroid, item {item_id}) = {margins['text_similarity']:.4f} -> textual/description compatibility metric")
    else:
        ev.append(f"description_feature.pt: missing text embedding for item {item_id} or centroid unavailable -> no textual similarity")
    # Also include item title from item_info.json if available (one line)
    try:
        with open(ITEM_INFO_PATH, "r", encoding="utf-8") as f:
            item_info = json.load(f)
    except:
        item_info = {}
    title = item_info.get(str(item_id), {}).get("title", "")
    if title:
        ev.append(f"item_info.json: item {item_id} title -> '{title}'")
    # Deduplicate and limit to 5
    unique = []
    for e in ev:
        if e not in unique:
            unique.append(e)
    return unique[:5]

def main():
    sources = load_sources()
    partial_bundle_context = retrieve_partial_bundle_context(PARTIAL_ITEMS, sources)
    candidate_evidence = {}
    for candidate in CANDIDATES:
        candidate_evidence[candidate["label"]] = {
            "item_id": candidate["item_id"],
            "evidence": retrieve_candidate_evidence(candidate, PARTIAL_ITEMS, partial_bundle_context, sources),
        }
    out = {
        "schema_version": "adaptive_bundle_evidence_v2",
        "intent": INTENT,
        "strategy": STRATEGY,
        "partial_bundle_evidence": {
            "evidence": partial_bundle_context["evidence"]
        },
        "candidate_evidence": candidate_evidence
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if name == "main":
    main()
