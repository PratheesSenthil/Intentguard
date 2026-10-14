import os

def test_dashboard_security():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dashboard_html = os.path.join(base_dir, "agent", "dashboard.html")
    dashboard_js = os.path.join(base_dir, "agent", "static", "dashboard.js")
    
    bad_terms = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("]
    
    for fpath in [dashboard_html, dashboard_js]:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            for term in bad_terms:
                assert term not in content, f"Found insecure DOM API '{term}' in {fpath}"
                
            # Check for inline scripts/styles in HTML
            if fpath.endswith(".html"):
                assert "<script>" not in content
                assert "<style>" not in content
