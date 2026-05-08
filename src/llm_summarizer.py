from transformers import pipeline
import torch

class LLMSummarizer:
    def __init__(self, model_name="google/flan-t5-small"):
        # Use a lightweight summarization model suitable for CPU execution
        print(f"Loading LLM model: {model_name}")
        self.device = 0 if torch.cuda.is_available() else -1
        try:
             self.summarizer = pipeline("summarization", model=model_name, device=self.device)
        except Exception as e:
             print(f"Warning: Could not load summarizer model. Error: {e}")
             self.summarizer = None

    def summarize_text(self, text, max_length=150, min_length=30):
        """Summarizes the input text."""
        if not self.summarizer:
             return "Model not loaded. Summarization disabled."
             
        if len(text.split()) < min_length:
            return "Text is too short to summarize."
            
        try:
             # Ensure input length is not too long for the model
             truncated_text = " ".join(text.split()[:512]) 
             summary = self.summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=False)
             return summary[0]['summary_text']
        except Exception as e:
             return f"Error during summarization: {e}"

if __name__ == "__main__":
    llm = LLMSummarizer()
    text = "The quick brown fox jumps over the lazy dog. The dog woke up and started barking at the fox. The fox ran away into the woods."
    print("Summary:", llm.summarize_text(text))
