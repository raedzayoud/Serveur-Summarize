from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
import logging
nltk.download('punkt')
nltk.download('punkt_tab')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def summarize_text(text, sentence_count=1, algorithm="textrank"):
    """
    Summarize the given text using the specified algorithm.
    
    Args:
        text (str): The text to summarize
        sentence_count (int): Number of sentences in the summary
        algorithm (str): The summarization algorithm to use (lsa or textrank)
        
    Returns:
        str: The summarized text
    """
    # Create parser
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Get the summarizer
    language = "english"
    stemmer = Stemmer(language)
    
    if algorithm.lower() == "lsa":
        summarizer = LsaSummarizer(stemmer)
    else:  # Default to TextRank for better single-sentence summaries
        summarizer = TextRankSummarizer(stemmer)
    
    # Apply stop words
    summarizer.stop_words = get_stop_words(language)
    
    # Generate summary
    summary = summarizer(parser.document, sentence_count)
    
    # Join sentences into a single string
    return " ".join(str(sentence) for sentence in summary)

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint to summarize text"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON in request body"}), 400
        
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Get optional parameters with defaults
        sentence_count = data.get('sentence_count', 1)  # Default to 1 for paragraph to single sentence
        algorithm = data.get('algorithm', 'textrank')  # TextRank works better for single-sentence summaries
        
        # Ensure sentence_count is a positive integer
        try:
            sentence_count = int(sentence_count)
            if sentence_count <= 0:
                raise ValueError("Sentence count must be a positive integer")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Validate text
        if not text or len(text.strip()) == 0:
            return jsonify({"error": "Text cannot be empty"}), 400
            
        # Generate summary
        summary = summarize_text(text, sentence_count, algorithm)
        
        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
