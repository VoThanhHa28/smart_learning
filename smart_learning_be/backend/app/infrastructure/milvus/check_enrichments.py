import os
import json

enrichments_dir = "././data/enrichments"

if not os.path.exists(enrichments_dir):
    print(f"❌ Directory not found: {enrichments_dir}")
else:
    files = os.listdir(enrichments_dir)
    print(f"📂 Found {len(files)} enrichment files:\n")

    for filename in files:
        filepath = os.path.join(enrichments_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        toc = data.get('toc', [])
        print(f"📄 {filename}")
        print(f"   TOC items: {len(toc)}")

        if toc:
            print(f"   Sample (first 3):")
            for i, item in enumerate(toc[:3], 1):
                print(f"      {i}. {item.get('title', 'N/A')} (page {item.get('page', 'N/A')})")
        else:
            print(f"   ⚠️ Empty TOC")
        print()