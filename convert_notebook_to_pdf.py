#!/usr/bin/env python
"""Convert Jupyter notebook to PDF via HTML"""
import json
from pathlib import Path

def convert_notebook_to_html(notebook_path, output_path=None):
    """Convert a Jupyter notebook to HTML (can be printed to PDF)"""
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract markdown cells
    markdown_content = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            # Handle both list and string formats
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            if content.strip():  # Only add non-empty cells
                markdown_content.append(content)
    
    # Combine all markdown content
    full_content = '\n\n'.join(markdown_content)
    
    # Create HTML with proper styling for PDF printing
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Homework Explanations</title>
    <style>
        @media print {
            @page {
                margin: 1in;
            }
            body {
                margin: 0;
            }
        }
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        p {
            margin: 1em 0;
            text-align: justify;
        }
        /* MathJax styling */
        .MathJax, .math {
            font-size: 1.1em;
        }
        /* Table styling */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
</head>
<body>
{content}
</body>
</html>"""
    
    # Convert markdown to HTML while preserving LaTeX
    # Use a simpler approach: convert line by line, preserving LaTeX
    import re
    import html
    
    def process_paragraph(text):
        """Process a paragraph, preserving LaTeX math"""
        if not text.strip():
            return ""
        
        # Escape HTML but preserve LaTeX math expressions
        # First, find and temporarily replace LaTeX
        math_expressions = []
        placeholder_pattern = "___MATH_EXPR_{}___"
        
        def replace_math(match):
            math_expressions.append(match.group(0))
            return placeholder_pattern.format(len(math_expressions) - 1)
        
        # Protect display math ($$...$$)
        protected = re.sub(r'\$\$.*?\$\$', replace_math, text, flags=re.DOTALL)
        # Protect inline math ($...$) - be careful not to match $$ as $
        protected = re.sub(r'(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)', replace_math, protected)
        
        # Now escape HTML
        protected = html.escape(protected)
        
        # Restore math expressions (they're already properly formatted)
        for i, math_expr in enumerate(math_expressions):
            protected = protected.replace(placeholder_pattern.format(i), math_expr)
        
        # Basic markdown: **bold**
        protected = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', protected)
        
        # Convert line breaks to <br>
        protected = protected.replace('\n', '<br>\n')
        
        return f'<p>{protected}</p>'
    
    # Split into paragraphs and process
    paragraphs = full_content.split('\n\n')
    html_paragraphs = [process_paragraph(p) for p in paragraphs if p.strip()]
    html_content = '\n'.join(html_paragraphs)
    
    # Wrap in HTML template (use replace instead of format to avoid issues with { in math)
    final_html = html_template.replace('{content}', html_content)
    
    if output_path is None:
        output_path = notebook_path.with_suffix('.html')
    else:
        output_path = Path(output_path)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"Successfully created HTML file: {output_path}")
    
    # Try to convert to PDF directly with LaTeX rendering
    pdf_path = output_path.with_suffix('.pdf')
    pdf_created = False
    
    # Try playwright first (renders MathJax properly)
    try:
        from playwright.sync_api import sync_playwright
        import http.server
        import socketserver
        import threading
        import time
        import os
        
        # Start a local HTTP server (MathJax doesn't work well with file://)
        PORT = 8000
        os.chdir(output_path.parent)
        
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", PORT), handler)
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        html_filename = output_path.name
        url = f"http://localhost:{PORT}/{html_filename}"
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            
            # Wait for MathJax to load and render
            page.wait_for_timeout(3000)  # Initial wait for MathJax to load
            
            # Wait for MathJax to finish rendering
            try:
                # Wait for MathJax to be defined
                page.wait_for_function("typeof MathJax !== 'undefined'", timeout=10000)
                
                # Wait for MathJax 3.x to finish typesetting
                page.wait_for_function("""
                    () => {
                        if (typeof MathJax === 'undefined') return false;
                        // MathJax 3.x API
                        if (MathJax.startup && MathJax.startup.document) {
                            try {
                                const state = MathJax.startup.document.state();
                                return state === 0; // STATE.READY = 0
                            } catch(e) {
                                // Fallback: check processing elements
                                return document.querySelectorAll('.MathJax_Processing, [class*="MathJax_Processing"]').length === 0;
                            }
                        }
                        // Fallback: check if there are no more processing elements
                        return document.querySelectorAll('.MathJax_Processing, [class*="MathJax_Processing"]').length === 0;
                    }
                """, timeout=20000)
            except Exception as e:
                # If detection fails, wait longer - MathJax should still render
                print(f"MathJax state detection had issues, using extended wait: {e}")
                page.wait_for_timeout(10000)
            
            # Final wait to ensure all rendering is complete
            page.wait_for_load_state('networkidle')
            page.wait_for_timeout(3000)
            
            page.pdf(path=str(pdf_path), format='A4', margin={'top': '1in', 'right': '1in', 'bottom': '1in', 'left': '1in'})
            browser.close()
        
        # Shutdown the server
        httpd.shutdown()
        
        print(f"Successfully created PDF file with LaTeX rendering: {pdf_path}")
        pdf_created = True
        return pdf_path
    except ImportError:
        print("Playwright not available. Please install it with: pip install playwright")
        print("Then run: python -m playwright install chromium")
    except Exception as e:
        print(f"Playwright PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to xhtml2pdf (no LaTeX rendering)
    if not pdf_created:
        try:
            from xhtml2pdf import pisa
            with open(pdf_path, 'wb') as pdf_file:
                pisa_status = pisa.CreatePDF(final_html, dest=pdf_file)
            if not pisa_status.err:
                print(f"Created PDF file (LaTeX may not render): {pdf_path}")
                print("Note: LaTeX equations may not be rendered properly.")
                pdf_created = True
                return pdf_path
        except ImportError:
            pass
        except Exception as e:
            print(f"xhtml2pdf failed: {e}")
    
    # If all failed, provide manual instructions
    if not pdf_created:
        print("\nCould not create PDF automatically.")
        print("To convert to PDF with LaTeX rendering:")
        print("1. Open the HTML file in your browser")
        print("2. Wait for MathJax to finish rendering (equations should appear)")
        print("3. Press Ctrl+P (or Cmd+P on Mac)")
        print("4. Select 'Save as PDF' as the destination")
        print("5. Click Save")
    
    return output_path

if __name__ == "__main__":
    notebook_path = "Homework 1/explanations.ipynb"
    output_path = "Homework 1/explanations.html"
    
    try:
        convert_notebook_to_html(notebook_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

