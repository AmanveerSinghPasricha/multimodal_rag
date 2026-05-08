import fitz  # PyMuPDF
import os
import hashlib
import logging
import base64
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, vision_llm, upload_dir="uploads/images"):
        """
        vision_llm: A multimodal model (e.g., Llama-3.2-Vision or Llama-4-Scout).
        """
        self.vision_llm = vision_llm
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def _encode_image_to_base64(self, image_path: str):
        """Helper to convert local images to LLM-ready format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def summarize_standalone_image(self, image_path: str):
        """Processes a standalone file uploaded by the user."""
        base64_image = self._encode_image_to_base64(image_path)
        
        prompt = [
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image in technical detail for a search index. Include all text, data points, and labels."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
        ]
        res = await self.vision_llm.ainvoke(prompt)
        return res.content

    async def summarize_pdf_image(self, image_path: str, page_num: int):
        """Analyzes a figure extracted from a specific PDF page."""
        try:
            base64_image = self._encode_image_to_base64(image_path)
            
            prompt = [
                HumanMessage(content=[
                    {
                        "type": "text", 
                        "text": f"This figure is from Page {page_num}. Analyze its technical significance, "
                                f"summarizing charts, tables, or diagrams for a search index."
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            res = await self.vision_llm.ainvoke(prompt)
            return res.content
        except Exception as e:
            logger.error(f"Vision failure on page {page_num}: {str(e)}")
            return f"[Technical Figure on Page {page_num}]"

    def extract_images_from_pdf(self, pdf_path: str):
        """Extracts and saves images from PDF; returns metadata list."""
        doc = fitz.open(pdf_path)
        img_metadata = []
        
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Unique ID based on image content to prevent redundant storage
                img_hash = hashlib.md5(image_bytes).hexdigest()
                filename = f"page_{page_index+1}_idx{img_idx}_{img_hash}.{ext}"
                save_path = os.path.join(self.upload_dir, filename)
                
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(image_bytes)
                
                img_metadata.append({
                    "path": save_path, 
                    "page": page_index + 1,
                    "universal_citation": f"{os.path.basename(pdf_path)} (Figure on Page {page_index + 1})"
                })
                
        doc.close()
        logger.info(f"Extracted {len(img_metadata)} images from {pdf_path}")
        return img_metadata