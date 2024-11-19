import gradio as gr
from PIL import Image
import os
from bharatOCR.ocr import OCR  # Ensure OCR class is saved in a file named ocr.py
from bharatOCR.theme import Seafoam

# Initialize the OCR object for text detection and recognition
ocr = OCR(verbose=False)

def process_image(image):
    """
    Processes the uploaded image for text detection and recognition. 
    - Detects bounding boxes in the image
    - Draws bounding boxes on the image and identifies script in each detected area
    - Recognizes text in each cropped region and returns the annotated image and recognized text

    Parameters:
    image (PIL.Image): The input image to be processed.

    Returns:
    tuple: A PIL.Image with bounding boxes and a string of recognized text.
    """
    
    # Save the input image temporarily
    image_path = "input_image.jpg"
    image.save(image_path)
    
    # Detect bounding boxes on the image using OCR
    detections = ocr.detect(image_path)
    
    # Draw bounding boxes on the image and save it as output
    ocr.visualize_detection(image_path, detections, save_path="output_image.png")
    
    # Load the annotated image with bounding boxes drawn
    output_image = Image.open("output_image.png")
    
    # Initialize list to hold recognized text from each detected area
    recognized_texts = []
    pil_image = Image.open(image_path)
    
    # Process each detected bounding box for script identification and text recognition
    for bbox in detections:
        # Identify the script and crop the image to this region
        script_lang, cropped_path = ocr.crop_and_identify_script(pil_image, bbox)
        
        if script_lang:  # Only proceed if a script language is identified
            # Recognize text in the cropped area
            recognized_text = ocr.recognise(cropped_path, script_lang)
            recognized_texts.append(recognized_text)
    
    # Combine recognized texts into a single string for display
    recognized_texts_combined = " ".join(recognized_texts)
    return output_image, recognized_texts_combined

# Custom HTML for interface header with logos and alignment
interface_html = """
<div style="text-align: center;">
    <img src="https://iitj.ac.in/uploaded_docs/IITJ%20Logo__big.jpg" alt="Logo" style="width: 100px; height: 100px; float: left;">
    <img src="https://play-lh.googleusercontent.com/_FXSr4xmhPfBykmNJvKvC0GIAVJmOLhFl6RA5fobCjV-8zVSypxX8yb8ka6zu6-4TEft=w240-h480-rw" alt="Right Image" style="width: 100px; height: 100px; float: right;">
</div>
"""


# Links to GitHub and Dataset repositories
links_html = """
<div style="text-align: center; padding-top: 20px;">
    <a href="https://github.com/your-github-repo" target="_blank" style="margin-right: 20px; font-size: 18px;">GitHub Repository</a>
    <a href="https://github.com/Bhashini-IITJ/BharatSceneTextDataset" target="_blank" style="font-size: 18px;">Dataset Repository</a>
</div>
"""
# Links to GitHub and Dataset repositories with GitHub icon
links_html = """
<div style="text-align: center; padding-top: 20px;">
    <a href="https://github.com/Bhashini-IITJ/BharatOCR" target="_blank" style="margin-right: 20px; font-size: 18px; text-decoration: none;">
        GitHub Repository
    </a>
    <a href="https://github.com/Bhashini-IITJ/BharatSceneTextDataset" target="_blank" style="font-size: 18px; text-decoration: none;">
        Dataset Repository
    </a>
</div>
"""

# Custom CSS to style the text box font size
custom_css = """
.custom-textbox textarea {
    font-size: 20px !important;
}
"""

# Create an instance of the Seafoam theme for a consistent visual style
seafoam = Seafoam()

# Define examples for users to try out
examples = [
    ["demo_images/image_141.jpg"],
    ["demo_images/image_1296.jpg"]
]

# Set up the Gradio Interface with the defined function and customizations
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=[
        gr.Image(type="pil", label="Detected Bounding Boxes"),
        gr.Textbox(label="Recognized Text", elem_classes="custom-textbox")
    ],
    title="BharatOCR - Indic Scene Text Recogniser Toolkit",
    description=interface_html+links_html,
    theme=seafoam,
    css=custom_css,
    examples=examples
)

# Server setup and launch configuration
if __name__ == "__main__":
    server = "0.0.0.0"  # IP address for server
    port = 7865  # Port to run the server on
    demo.launch(server_name=server, server_port=port)
