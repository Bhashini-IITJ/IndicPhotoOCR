import json

# Load the JSON file
with open('test_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Simulated predicted transcriptions (replace with your OCR output)
predicted_transcriptions = {
    "locality_01pic.jpg418574": ["मा", "", "", "", "", "", "", "", ""],  # Example with errors
    "locality_02pic.jpg250234": ["सत्यमेव", "जयते", "नगरर", "पंचाय", "बिहिया", "भोजपुर", "प्रशासनीक", "भवन", "स्थापित"],
    "locality_03pic.jpg237342": ["विश्वकर्मा", "जनरल", "हार्डवेयर", "स्टेशन", "रोड", "बिहिया", "भोजपुर", "बिहार", "802152", "greenply", "पेंट", "दरवाजा", "हार्डवेयर", "शीशा", "पलाई", "माइका", "इटायादी", "NEROLAC"],
    "locality_04pic.jpg132766": ["SBI", "भारती", "स्टेट बैंक", "बिहिया", "शाखा", "State", "Bank", "OF INDIA", "BEHEA", "BRANCH"],
    "locality_05pic.jpg137625": ["बिहिया", "मई", "महथिन", "मई मंदिर", "दूरी", "मखदूम", "बाबा का मazaar", "दूरी", "वीर", "कुंवर सिंह", "जन्मभूमि", "जगदीशपुर", "दूरी", "जाने", "लिए यहाँ उतरिए", "BIHIYA", "Ht.ABOVE"]
}

# Function to calculate WRR
def calculate_wrr(ground_truth, predicted):
    if not ground_truth or not predicted:
        return 0.0
    correct_words = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == pred and gt.strip())
    total_words = len([gt for gt in ground_truth if gt.strip()])
    return (correct_words / total_words * 100) if total_words > 0 else 0.0

# Process each image
for image_id, image_data in data.items():
    regions = image_data['regions']
    ground_truth = [region['region_attributes'].get('Transcription', '') if region else '' for region in regions]
    predicted = predicted_transcriptions.get(image_id, ['' for _ in regions])[:len(regions)]
    
    print(f"\nImage: {image_id}")
    print("Ground Truth:", ground_truth)
    print("Predicted:", predicted)
    wrr = calculate_wrr(ground_truth, predicted)
    print(f"Word Recognition Rate (WRR): {wrr:.2f}%")

    # Print individual transcriptions
    for region in regions:
        if region and 'region_attributes' in region:
            print(region['region_attributes'].get('Transcription', 'No transcription'))