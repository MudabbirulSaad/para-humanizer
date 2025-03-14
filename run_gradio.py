#!/usr/bin/env python
"""
Gradio web interface for Para-Humanizer.
Provides an easy-to-use browser-based UI for paraphrasing text.
"""
import os
import gradio as gr
import argparse
import logging
from api.config import load_settings
from para_humanizer.core.paraphraser import UltimateParaphraser
from para_humanizer.utils.text_analyzer import text_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("para-humanizer-gradio")

def create_paraphraser(config_file=None):
    """Initialize the paraphraser with configuration."""
    # Load configuration
    settings = load_settings(config_file)
    if config_file and os.path.exists(config_file):
        logger.info(f"Loaded configuration from {config_file}")
    
    # Initialize paraphraser
    paraphraser = UltimateParaphraser(
        use_gpu=settings.paraphraser.use_gpu,
        batch_size=settings.paraphraser.batch_size
    )
    
    return paraphraser

def paraphrase_text(
    text, 
    use_intelligent_params=True,
    rule_based_rate=0.4, 
    transformer_rate=0.0,
    humanize=True,
    humanize_intensity=0.5,
    typo_rate=0.01,
    preserve_structure=True
):
    """Process text with the paraphraser using provided parameters."""
    if not text.strip():
        return "Please enter some text to paraphrase.", ""
    
    try:
        # Get parameters either from intelligent selection or manual input
        if use_intelligent_params:
            # Analyze text to determine optimal parameters
            analysis_results = text_analyzer.analyze(text)
            
            # Map the returned keys from the analyze method to our expected keys
            effective_rule_based = analysis_results["rule_based_rate"]
            effective_transformer = analysis_results["transformer_rate"]
            effective_humanize = True  # Default to True as the analyze method doesn't return this
            effective_humanize_intensity = analysis_results["humanize_intensity"]
            effective_typo_rate = analysis_results["typo_rate"]
            
            # Build parameters description
            params_description = f"""
            ### Intelligent Parameter Selection
            Based on analysis of your text:
            
            * Word count: {len(text.split())}
            
            **Applied parameters:**
            * Rule-based rate: {effective_rule_based:.2f}
            * Transformer rate: {effective_transformer:.2f}
            * Humanize: {"Yes" if effective_humanize else "No"}
            * Humanize intensity: {effective_humanize_intensity:.2f}
            * Typo rate: {effective_typo_rate:.3f}
            * Preserve structure: {"Yes" if preserve_structure else "No"}
            """
        else:
            # Use manually specified parameters
            effective_rule_based = rule_based_rate
            effective_transformer = transformer_rate
            effective_humanize = humanize
            effective_humanize_intensity = humanize_intensity
            effective_typo_rate = typo_rate
            
            # Build parameters description
            params_description = f"""
            ### Manual Parameter Selection
            
            **Applied parameters:**
            * Rule-based rate: {effective_rule_based:.2f}
            * Transformer rate: {effective_transformer:.2f}
            * Humanize: {"Yes" if effective_humanize else "No"}
            * Humanize intensity: {effective_humanize_intensity:.2f}
            * Typo rate: {effective_typo_rate:.3f}
            * Preserve structure: {"Yes" if preserve_structure else "No"}
            """
        
        # Process the text with the paraphraser
        paraphrased = paraphraser_instance.paraphrase(
            text,
            rule_based_rate=effective_rule_based,
            transformer_rate=effective_transformer,
            humanize=effective_humanize,
            humanize_intensity=effective_humanize_intensity,
            typo_rate=effective_typo_rate,
            preserve_structure=preserve_structure
        )
        
        return paraphrased, params_description
    except Exception as e:
        logger.error(f"Error in paraphrasing: {e}", exc_info=True)
        return f"Error: {str(e)}", "An error occurred during paraphrasing."

def create_gradio_interface(paraphraser):
    """Create the Gradio web interface."""
    global paraphraser_instance
    paraphraser_instance = paraphraser
    
    # Style and theme - Use dark theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    
    # Custom CSS for rounded corners and other styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Main container and component styling */
    .gradio-row, .gradio-col, .gradio-accordion, .gradio-box {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Input and output textboxes */
    .gradio-textbox {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .gradio-textbox textarea {
        border-radius: 8px !important;
        padding: 12px !important;
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
    }
    
    /* Button styling */
    .gradio-button {
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        padding: 8px 16px !important;
    }
    
    .gradio-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Copy button styling */
    .copy-btn {
        margin-top: 5px !important;
        width: 100% !important;
    }
    
    /* Headings and text */
    .gradio-markdown h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
        color: rgb(59, 130, 246) !important;
    }
    
    .gradio-markdown h2 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        color: rgb(226, 232, 240) !important;
    }
    
    /* Checkbox and other controls */
    .gradio-checkbox {
        margin: 8px 0 !important;
    }
    """
    
    # Create the interface
    with gr.Blocks(theme=theme, title="Para-Humanizer", css=custom_css) as interface:
        gr.Markdown(
            """
            # Para-Humanizer
            
            **A smart paraphrasing tool that makes text sound more human**
            
            This tool uses advanced NLP techniques to paraphrase text while preserving meaning and adding human-like variations. 
            It combines rule-based rewriting, neural models, and randomized humanization.
            """
        )
        
        # Main content rows
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Original Text")
                input_text = gr.Textbox(
                    label="",
                    placeholder="Enter text here to paraphrase...",
                    lines=10
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        use_intelligent = gr.Checkbox(
                            label="Use Intelligent Parameter Selection",
                            value=True,
                            info="Automatically choose optimal parameters based on text analysis"
                        )
                    
                    with gr.Column(scale=2):
                        paraphrase_button = gr.Button("Paraphrase", variant="primary", size="lg")
                
                # Structure preservation option
                with gr.Row():
                    preserve_structure_check = gr.Checkbox(
                        label="Preserve Document Structure",
                        value=True,
                        info="Maintain original formatting, bullet points, and paragraph structure"
                    )
                
                # Advanced parameters (hidden by default when intelligent selection is enabled)
                with gr.Accordion("Advanced Parameters", open=False) as advanced_params:
                    with gr.Row():
                        rule_based = gr.Slider(
                            label="Rule-Based Rate",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.4,
                            step=0.05,
                            info="How much to apply rule-based changes (0-1)"
                        )
                        transformer = gr.Slider(
                            label="Transformer Rate",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            info="How much to apply neural model changes (0-1)"
                        )
                    
                    with gr.Row():
                        humanize_check = gr.Checkbox(
                            label="Humanize",
                            value=True,
                            info="Add human-like variations"
                        )
                        humanize_slider = gr.Slider(
                            label="Humanize Intensity",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            info="Intensity of humanization (0-1)"
                        )
                        typo_slider = gr.Slider(
                            label="Typo Rate",
                            minimum=0.0,
                            maximum=0.1,
                            value=0.01,
                            step=0.005,
                            info="Rate of typographical errors (0-0.1)"
                        )
            
            with gr.Column(scale=1):
                gr.Markdown("## Result")
                output_text = gr.Textbox(
                    label="",
                    lines=10
                )
                
                copy_button = gr.Button("📋 Copy to Clipboard", variant="secondary", elem_classes=["copy-btn"])
                
                gr.Markdown("## Parameters Used")
                params_text = gr.Markdown("")
        
        # Show/hide advanced parameters based on intelligent checkbox
        def update_advanced_visibility(use_intelligent):
            return gr.update(visible=not use_intelligent)
            
        use_intelligent.change(
            fn=update_advanced_visibility,
            inputs=[use_intelligent],
            outputs=[advanced_params]
        )
        
        # Set up the paraphrase function - make sure parameter order matches the paraphrase_text function
        paraphrase_button.click(
            fn=paraphrase_text,
            inputs=[
                input_text,
                use_intelligent,
                rule_based,
                transformer,
                humanize_check,
                humanize_slider,
                typo_slider,
                preserve_structure_check
            ],
            outputs=[output_text, params_text]
        )
        
        # Set up copy function
        def copy_text(text):
            """Copy text and show notification"""
            from pyperclip import copy
            try:
                copy(text)
                return text, "✓ Text copied to clipboard!"
            except Exception:
                return text, "➤ Please use Ctrl+C to copy text manually"
        
        copy_button.click(
            fn=copy_text,
            inputs=[output_text],
            outputs=[output_text, params_text]
        )
        
        # Examples with variety in text length and type
        examples = [
            ["The implementation of robust algorithmic solutions requires careful consideration of computational complexity."],
            ["Hey! Just wanted to let you know that I'm running a bit late. Traffic is crazy today!"],
            ["The system uses a distributed architecture with load balancing across multiple nodes."],
            ["Can you please send me the report ASAP? Thanks!"],
            ["""The Hidden Impact of Ocean Currents on Global Climate

Ocean currents, the continuous, directed movements of seawater, constitute one of Earth's most influential yet least visible natural phenomena. These massive flows of water function as planetary conveyor belts, transporting heat, nutrients, and marine life across thousands of miles and connecting disparate ecosystems. Their profound impact on global climate often goes unrecognized by the general public, yet oceanographers and climate scientists have long understood that these marine highways fundamentally shape weather patterns, temperature distributions, and even the habitability of entire continents."""],
            ["""Artificial intelligence has transformed from a theoretical concept into a cornerstone of modern society. Its journey began in the mid-20th century when pioneering computer scientists first contemplated the possibility of creating machines capable of mimicking human thought processes. These early visionaries, including Alan Turing, John McCarthy, and Marvin Minsky, laid the groundwork for what would eventually become one of the most transformative technological developments in human history."""]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=input_text
        )
        
        # Footer
        gr.Markdown(
            """
            ### About Para-Humanizer
            
            Para-Humanizer is an advanced text paraphrasing tool that combines rule-based techniques with neural models to create natural-sounding paraphrases. It can intelligently select parameters based on text analysis or allow manual control.
            
            **Key Features:**
            
            - **Intelligent Parameter Selection:** Automatically choose optimal parameters based on text analysis.
            - **Manual Control:** Fine-tune parameters to suit your specific needs.
            - **Rule-Based Techniques:** Apply rule-based changes to preserve meaning and context.
            - **Neural Models:** Utilize neural models to generate human-like variations.
            
            **License:** MIT License
            
            **Source Code:** [View on GitHub](https://github.com/mudabbirulsaad/para-humanizer)
            """
        )
    
    return interface

def main():
    """Main function to run the Gradio interface."""
    parser = argparse.ArgumentParser(description="Run Para-Humanizer Gradio interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    logger.info("Initializing Para-Humanizer Gradio interface")
    paraphraser = create_paraphraser(args.config)
    
    interface = create_gradio_interface(paraphraser)
    
    share_info = ""
    if args.share:
        share_info = "Creating a public share link... this may take a moment."
        logger.info("Creating a public share link")
    else:
        share_info = "For a shareable URL, restart with the --share flag: python run_gradio.py --share"
        logger.info("Starting without public share link")
    
    print("\n" + "="*80)
    print(f"Para-Humanizer Gradio Interface")
    print("="*80)
    print(f"Local URL: http://{args.host}:{args.port}")
    print(share_info)
    print("="*80 + "\n")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        favicon_path="docs/images/favicon.ico"
    )

if __name__ == "__main__":
    main()
