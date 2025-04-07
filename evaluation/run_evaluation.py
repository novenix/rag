import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import RAGEvaluator  # Changed from relative to absolute import
from rag.retriever import TFIDFRetriever, DenseRetriever
from rag.generator import get_generator  # Add this import for the get_generator function
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument('--retriever', type=str, choices=['tfidf', 'dense'], default='tfidf',
                        help='Retriever type to use (tfidf or dense)')
    parser.add_argument('--test_file', type=str, 
                        default='evaluation/test_queries.json',
                        help='Path to test queries JSON file')
    args = parser.parse_args()
    
    # Initialize components based on arguments
    if args.retriever == 'tfidf':
        retriever = TFIDFRetriever()
    else:
        retriever = DenseRetriever()
    
    generator = get_generator(provider="together")
    
    # Initialize and run evaluator
    evaluator = RAGEvaluator(retriever, generator)
    results = evaluator.run_evaluation(args.test_file)
    evaluator.print_evaluation_report(results)
    
    # Save detailed results to file
    import json
    with open(f'evaluation_results_{args.retriever}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to evaluation_results_{args.retriever}.json")

if __name__ == "__main__":
    main()

## python -m evaluation.run_evaluation --retriever tfidf
## o 
## ## python -m run_evaluation --retriever tfidf
#dense
#python -m evaluation.run_evaluation --retriever dense
