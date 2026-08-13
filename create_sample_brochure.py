"""
create_sample_brochure.py
Creates a realistic sample PDF brochure in brochure/ folder using PyMuPDF (fitz) or reportlab.
"""

import os

def generate_brochure_pdf():
    os.makedirs("brochure", exist_ok=True)
    pdf_path = os.path.join("brochure", "Galactic_3D_Manufacturing_Capabilities.pdf")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open()

        # Page 1: Overview & Rapid Prototyping
        page1 = doc.new_page()
        p1_text = """
GALACTIC 3D MANUFACTURING & ENGINEERING SOLUTIONS
Official Company Brochure & Capability Overview

1. ABOUT GALACTIC 3D
Galactic 3D is a premier provider of high-precision manufacturing, rapid prototyping, and industrial engineering solutions.
We serve global leaders across Automotive, Aerospace, Medical Devices, Defense, Heavy Engineering, and Industrial Automation.

2. ADDITIVE MANUFACTURING & 3D PRINTING
- High-resolution 3D Printing (SLA, SLS, DMLS, FDM)
- Rapid Prototyping for functional testing & low-volume production
- Metal 3D Printing in Stainless Steel, Titanium, and Aluminium alloys
- Reverse Engineering and CAD/CAM design optimization
        """
        page1.insert_text((50, 50), p1_text.strip(), fontsize=11)

        # Page 2: CNC Machining, Sheet Metal & Tool Room
        page2 = doc.new_page()
        p2_text = """
GALACTIC 3D CAPABILITY PORTFOLIO (Page 2)

3. PRECISION CNC MACHINING & TOOL ROOM
- 5-Axis CNC Milling, Turning, and Wire EDM
- Precision Components with tolerances down to +/- 0.005mm
- Comprehensive Tool Room capabilities for Jigs, Fixtures, and Custom Tooling
- Quality Inspection using Coordinate Measuring Machines (CMM) and Metrology

4. SHEET METAL FABRICATION & STAMPING
- High-speed Laser Cutting (up to 20mm steel sheet thickness)
- Precision Bending, Metal Stamping, Enclosures, and Robotic Welding
- Surface Finishing: Anodizing, Powder Coating, Electroplating, and Heat Treatment

5. INJECTION MOULDING & CASTING
- Plastic Injection Moulding with Cleanroom manufacturing for Medical Devices
- Die Casting, Sand Casting, and Investment Casting for heavy automotive parts
- Rapid Tooling and Mold Flow Analysis
        """
        page2.insert_text((50, 50), p2_text.strip(), fontsize=11)

        doc.save(pdf_path)
        doc.close()
        print(f"Successfully generated sample brochure at: {pdf_path}")
    except Exception as e:
        print(f"Error creating brochure PDF: {e}")

if __name__ == "__main__":
    generate_brochure_pdf()
