import os

from dotenv import load_dotenv
from google import genai
from metaflow import FlowSpec, Parameter, card, step


class Assignment9(FlowSpec):
    """Call an LLM with a prompt and visualize the response in a card.

    Flow: start -> generate -> end
    The 'generate' step sends the prompt to Gemini and stores the
    response as an artifact. The card renders the pair in a styled layout.
    """

    prompt = Parameter(
        "prompt",
        help="Text prompt to send to the LLM",
        default="Explain what Metaflow is in two sentences.",
    )

    @step
    def start(self):
        """Log the prompt and proceed."""
        print(f"Prompt: {self.prompt}")
        self.next(self.generate)

    @card(type="html")
    @step
    def generate(self):
        """Call Gemini and store the response as an artifact."""
        load_dotenv()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.prompt,
        )
        self.response = response.text
        print(f"Response: {self.response}")

        self.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: sans-serif;
                    max-width: 720px;
                    margin: 40px auto;
                    padding: 0 24px;
                    color: #333;
                }}
                .label {{
                    font-size: 0.75em;
                    font-weight: bold;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    color: #888;
                    margin-bottom: 6px;
                }}
                .prompt-box {{
                    background: #f0f4ff;
                    border-left: 4px solid #4a6cf7;
                    padding: 16px 20px;
                    border-radius: 4px;
                    margin-bottom: 24px;
                    font-size: 1.1em;
                }}
                .response-box {{
                    background: #f6fef6;
                    border-left: 4px solid #34a853;
                    padding: 16px 20px;
                    border-radius: 4px;
                    line-height: 1.7;
                    font-size: 1.05em;
                    white-space: pre-wrap;
                }}
                h1 {{ color: #1a1a2e; margin-bottom: 32px; }}
            </style>
        </head>
        <body>
            <h1>LLM Prompt & Response</h1>

            <div class="label">Prompt</div>
            <div class="prompt-box">{self.prompt}</div>

            <div class="label">Response</div>
            <div class="response-box">{self.response}</div>
        </body>
        </html>
        """
        self.next(self.end)

    @step
    def end(self):
        """Print the final prompt/response pair."""
        print(f"\nPrompt:   {self.prompt}")
        print(f"Response: {self.response}")


if __name__ == "__main__":
    Assignment9()
