import os, json, re, requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd

INDEX_URL = "https://crisprmedicinenews.com/clinical-trials/"

def get_first_n_nct_urls(n=250):
    r = requests.get(INDEX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pat = re.compile(r"NCT\d+", re.IGNORECASE)
    urls = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if pat.search(text):
            urls.append(urljoin(INDEX_URL, a["href"]))
            if len(urls) >= n:
                break
    if not urls:
        raise RuntimeError("No NCT links found on index page")
    return urls

def fetch_page_text(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def extract_trial_json(url):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    page_text = fetch_page_text(url)[:14000]
    system = "You are a precise scientific extraction assistant. Extract clinical trial info from noisy web page text. Return ONLY strict JSON with the requested keys."
    user = f"""From the page below, return ONLY JSON with exactly these keys:
    {{"study": string or null, "disease_category": string or null, "enrollment_target": int or null,
    "countries": string or null, "in_us": string or null, "start_year": int or null,
    "last_updated_year": int or null, "gene_editing_method": string or null,
    "phase_1": int or null, "phase_2": int or null, "phase_3": int or null}}
    Page URL: {url}
    Page text: {page_text}"""
    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fix = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            messages=[{"role": "system", "content": "Return valid JSON only."},
                      {"role": "user", "content": f"Make this valid JSON:\n\n{content}"}])
        return json.loads(fix.choices[0].message.content.strip())

def collect_trials(n=250):
    os.environ["OPENAI_API_KEY"] = input("Paste your OpenAI API key: ").strip()
    trial_urls = get_first_n_nct_urls(n=n)
    rows = []
    for url in trial_urls:
        print("Processing:", url)
        data = extract_trial_json(url)
        rows.append({
            "trial_url": url,
            "Study (disease • registry)": data.get("study"),
            "Disease Category": data.get("disease_category"),
            "Enrollment (target)": data.get("enrollment_target"),
            "Countries (per CMN page)": data.get("countries"),
            "In US?": data.get("in_us"),
            "Start year": data.get("start_year"),
            "Last-updated year": data.get("last_updated_year"),
            "Gene-editing method": data.get("gene_editing_method"),
            "Phase 1": data.get("phase_1"),
            "Phase 2": data.get("phase_2"),
            "Phase 3": data.get("phase_3"),
        })
    df = pd.DataFrame(rows)
    df.to_csv("CRISPR_Data_raw.csv", index=False)
    print("Saved to CRISPR_Data_raw.csv")
    return df
