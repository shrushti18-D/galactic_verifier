import time
import pandas as pd
from cleaner import detect_column_mapping, clean_dataset
from brochure_reader import read_all_brochures, extract_capabilities
from analyzer import analyze_companies_batch

def benchmark_9000():
    print("Generating 9,090 test company rows...")
    categories = ["CNC Machining", "Sheet Metal Fabrication", "Medical Injection Moulding", "Aerospace Engineering", "Digital Marketing", "Organic Farming", "Software Development", "Hotel Hospitality"]
    keywords_sample = ["CNC milling, turning, aerospace", "laser cutting, sheet metal bending", "cleanroom injection moulding", "3D printing, titanium casting", "SEO, Google ads", "fresh apples", "Python microservices", "luxury hotel room"]

    data = []
    for i in range(9090):
        c_idx = i % len(categories)
        data.append({
            "co_name": f"Company {i+1} Corp",
            "category": categories[c_idx],
            "keywords": keywords_sample[c_idx],
            "website": f"company{i+1}.com",
            "email": f"contact@company{i+1}.com"
        })

    df = pd.DataFrame(data)
    mapping = detect_column_mapping(df)
    cleaned_df, stats = clean_dataset(df, mapping)
    b_info = read_all_brochures("brochure")
    profile = extract_capabilities(b_info["combined_text"])

    start_time = time.time()
    print("Starting ML analysis on 9,090 rows...")
    res_df = analyze_companies_batch(cleaned_df, mapping, profile, batch_size=256)
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Analyzed {len(res_df):,} companies in ONLY {elapsed:.2f} seconds!")

if __name__ == "__main__":
    benchmark_9000()
