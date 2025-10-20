import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import os
from collections import OrderedDict
import matplotlib.pyplot as plt


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # TODO: Implement single-head attention
        # Create query, key, value, and output projection layers
        # Remember to set bias=False for query, key, value projections. 
        # Example : self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, z, past_kv=None, use_cache=False):
         # TODO: Implement the forward pass for single-head attention
        # 1. Compute query, key, value projections
        # 2. Handle KV caching if use_cache=True
        # 3. Compute attention scores with scaling #hint use torch.bmm
        # 4. Apply causal masking #hint: use torch.triu
        # 5. Apply softmax to get attention weights
        # 6. Compute context vector #hint: use torch.bmm
        # 7. Compute output and return
        
        # Your code here
        batch_size, seq_len, _ = z.shape
        
        key = self.W_k(z)
        value = self.W_v(z)
        query = self.W_q(z)
        
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            key = torch.cat([past_k, key], dim=1)
            value = torch.cat([past_v, value], dim=1)
        
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.d_model ** 0.5)
        
        # Create causal mask accounting for past_kv
        if past_kv is not None:
            past_len = past_kv[0].shape[1]
        else:
            past_len = 0
            
        mask = torch.ones(seq_len, seq_len + past_len, dtype=torch.bool, device=z.device)
        mask = torch.triu(mask, diagonal=1)
        
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, value)
        output = self.W_o(context)
        
        new_kv = (key, value) if use_cache else None
        # Placeholder return - replace with your implementation
        return output, new_kv

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Implement feed-forward network
        # Create two linear layers 
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Your code here
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, z):
        # TODO: Implement the forward pass for feed-forward network
        # Two linear layers with ReLU activation in between
        # Your code here
        z = self.fc1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        # Placeholder return - replace with your implementation
        return z

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement decoder layer
        # Create attention, feed-forward, layer norms, and dropout
        
        # Your code here
        self.attention = SingleHeadAttention(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder layer
        # 1. Apply layer norm before attention
        # 2. Apply attention followed by dropout
        # 3. Apply layer norm before feed-forward
        # 4. Apply feed-forward then residual connection
        
        # Your code here
        norm_z = self.norm1(z)
        attn_out, new_kv = self.attention(norm_z, past_kv, use_cache)
        z = z + self.dropout(attn_out)
        
        # Feed-forward block
        norm_z = self.norm2(z)
        ffn_out = self.feed_forward(norm_z)
        z = z + self.dropout(ffn_out)
        
        # Placeholder return - replace with your implementation
        return z, new_kv


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, past_kv=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        
        token_embeds = self.token_embedding(input_ids)

        # Handling position offsets for cached generation
        position_offset = past_kv[0][0].shape[1] if (past_kv and past_kv[0]) else 0

        position_ids = torch.arange(position_offset, position_offset + seq_len, 
                                    dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_ids = torch.clamp(position_ids, min=0, max=self.max_seq_len - 1)

        pos_embeds = self.pos_embedding(position_ids)
        x = token_embeds + pos_embeds

        if past_kv is None:
            past_kv = [None] * self.num_layers

        new_past_kv = []
        for i, layer in enumerate(self.layers):
            x, layer_kv = layer(x, past_kv[i], use_cache)
            new_past_kv.append(layer_kv)

        x = self.norm(x)
        logits = self.output_projection(x)

        return logits, new_past_kv if use_cache else None

    # Generates new tokens autoregressively
    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_cache=True):
        
        generated_ids = input_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            logits, past_kv = self(generated_ids, past_kv, use_cache)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            generated_ids = torch.clamp(generated_ids, min=0, max=self.vocab_size - 1)

        return generated_ids

# ================= DO NOT MODIFY CODE BELOW THIS LINE =================
# Evaluation harness code - do not modify

def load_test_cases(filepath):
    """Load test cases from a file."""
    with open(filepath, 'r') as f:
        test_cases = json.load(f)
    
    # Convert lists back to tensors
    for case in test_cases:
        case['input_ids'] = torch.tensor(case['input_ids'])
        case['expected_logits_no_cache'] = torch.tensor(case['expected_logits_no_cache'])
        case['expected_logits_with_cache'] = torch.tensor(case['expected_logits_with_cache'])
        case['expected_logits_sequential'] = torch.tensor(case['expected_logits_sequential'])
    
    return test_cases

def evaluate_model(model, test_cases, atol=1e-1, with_kv=False):
    """Evaluate model against test cases."""
    model.eval()
    results = []
    
    for i, case in enumerate(test_cases):
        input_ids = case['input_ids']
        expected_logits_no_cache = case['expected_logits_no_cache']
        # expected_logits_with_cache = case['expected_logits_with_cache']
        # expected_logits_sequential = case['expected_logits_sequential']
        
        with torch.no_grad():
            # Test without caching
            logits_no_cache, _ = model(input_ids, use_cache=False)
            no_cache_match = torch.allclose(logits_no_cache, expected_logits_no_cache, atol=atol)
            
            result = {
                'test_case' : i + 1,
                'no_cache_match' : no_cache_match,
                'all_match' : no_cache_match
            }
            if with_kv:
                # Test with caching (full sequence)
                expected_logits_with_cache = case['expected_logits_with_cache']
                logits_with_cache, _ = model(input_ids, use_cache=True)
                with_cache_match = torch.allclose(logits_with_cache, expected_logits_with_cache, atol=atol)

                cache_nocache_match = torch.allclose(logits_no_cache, logits_with_cache, atol=atol)
            
        
        result.update({
            'with_cache_match': with_cache_match if with_kv else None,
            'cache_nocache_match': cache_nocache_match if with_kv else None,
            'all_match': no_cache_match and (with_cache_match and cache_nocache_match if with_kv else no_cache_match)
        })
        
        if not result['all_match']:
            # Calculate error metrics for debugging
            if not no_cache_match:
                result['no_cache_max_error'] = torch.max(torch.abs(logits_no_cache - expected_logits_no_cache)).item()
            if with_kv and not with_cache_match:
                result['with_cache_max_error'] = torch.max(torch.abs(logits_with_cache - expected_logits_with_cache)).item()
            if with_kv and not cache_nocache_match:
                result['cache_nocache_max_error'] = torch.max(torch.abs(logits_no_cache - logits_with_cache)).item()
            
        results.append(result)
    
    # Overall results
    all_passed = all(r['all_match'] for r in results)
    pass_rate = sum(r['all_match'] for r in results) / len(results)
    
    summary = {
        'all_passed': all_passed,
        'pass_rate': pass_rate,
        'num_test_cases': len(test_cases),
        'num_passed': sum(r['all_match'] for r in results),
        'detailed_results': results
    }
    
    return summary

def benchmark_performance(model, input_ids, num_new_tokens=20, use_cache=True, num_runs=3):
    """Benchmark model performance."""
    model.eval()
    
    # Warm-up run
    model.generate(input_ids, num_new_tokens, use_cache=use_cache)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(input_ids, num_new_tokens, use_cache=use_cache)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Transformer Evaluation Harness')
    parser.add_argument('--mode', type=str, default='run', choices=['generate', 'evaluate', 'kv_evaluate', 'benchmark','plot', 'run'], 
                        help='Mode to run in')
    parser.add_argument('--weights', type=str, default='reference_weights.pt', 
                        help='Path to weights file')
    parser.add_argument('--model_state_dict', type=str, default='model_state_dict.pt', 
                        help='Path to model state dictionary file')
    parser.add_argument('--test_cases', type=str, default='test_cases.json', 
                        help='Path to test cases file')
    parser.add_argument('--vocab_size', type=int, default=1000, 
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=50, 
                        help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=100, 
                        help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of decoder layers')
    parser.add_argument('--max_seq_len', type=int, default=128, 
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate evaluation harness -- not accessible to students
        generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
    
    elif args.mode == 'evaluate':
        # Evaluate a model
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=False)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        # Print result stats - each test case with pass/fail info
        #print(f"  Detailed results: {results['detailed_results']}")

        
        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")

    elif args.mode == 'kv_evaluate':
        # Evaluate a model with kv cache against no_kv_cache
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=True)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        #detailed results
        #print(f"  Detailed results: {results['detailed_results']}")

        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")
                    if not result.get('with_cache_match', True):
                        print(f"    With cache: Failed (max error: {result.get('with_cache_max_error', 'N/A')})")
    
    elif args.mode == 'benchmark':

        # Benchmark model performance
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        

        # Load model state dict
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return

        # Create sample input
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        # Benchmark with and without caching
        print("Benchmarking...")
        time_without_cache = benchmark_performance(model, input_ids, use_cache=True)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=False)
        
        print(f"Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_without_cache / time_with_cache:.2f}x")
        
    elif args.mode == 'plot':
        #for plotting the graph 
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
    
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return

        # Sequence lengths to test
        sequence_lengths = [10, 50, 100, 500, 1000]
        latencies_without_cache = []
        latencies_with_cache = []

        print("Benchmarking...")
        for seq_len in sequence_lengths:
            input_ids = torch.randint(0, args.vocab_size, (1, seq_len))

            time_without_cache = benchmark_performance(model, input_ids, use_cache=True)
            time_with_cache = benchmark_performance(model, input_ids, use_cache=False)

            latencies_without_cache.append(time_without_cache)
            latencies_with_cache.append(time_with_cache)

            print(f"Seq Len: {seq_len}, Without KV cache: {time_without_cache:.4f}s, With KV cache: {time_with_cache:.4f}s")

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(sequence_lengths, latencies_without_cache, marker='o', linestyle='-', label='Without KV Cache')
        plt.plot(sequence_lengths, latencies_with_cache, marker='s', linestyle='-', label='With KV Cache')

        plt.xlabel("Sequence Length")
        plt.ylabel("Latency (seconds)")
        plt.title("Transformer Generation Latency vs. Sequence Length")
        plt.legend()
        plt.grid()
        plt.show()


    elif args.mode == 'run':
        # Just a debugging mode

        # Default mode: generate harness if files don't exist, then evaluate and benchmark
        if not os.path.exists('model_state_dict.pt') or not os.path.exists('test_cases.json'):
            generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Load model state dict
        state_dict = torch.load('model_state_dict.pt')

        # Print specific weights from state dict
        #print("From state dict - layer 0 Wq weight:")
        #print(state_dict['layers.0.attention.W_q.weight'])
        #print("From state dict - layer 1 Wk weight:")
        #print(state_dict['layers.1.attention.W_k.weight'])

        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model state dictionary from model_state_dict.pt")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
 
        print("Weights loaded from {}".format(args.model_state_dict))

        # print the structure of state_dict
        #print("State dict structure:")
        #print(state_dict.keys())

        # Verify they're the same
        print("Weights match for layer 0 Wq:", 
        torch.allclose(state_dict['layers.0.attention.W_q.weight'], model.layers[0].attention.W_q.weight))
        print("Weights match for layer 1 Wk:", 
        torch.allclose(state_dict['layers.1.attention.W_k.weight'], model.layers[1].attention.W_k.weight))
        
        # Load test cases
        test_cases = load_test_cases('test_cases.json')
        print(f"Test cases loaded from test_cases.json")

        # Print the first test case
        print("First test case:")
        print(test_cases[0].keys())
        # print the tensor shape for each key
        for key in test_cases[0].keys():
            print(key, test_cases[0][key].shape)

        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_model(model, test_cases)
        
        # Print evaluation results
        print(f"Evaluation Results:")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        
        # Benchmark
        print("\nBenchmarking performance...")
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Performance Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_with_cache > 0 and time_without_cache / time_with_cache:.2f}x")

if __name__ == "__main__":
    main()