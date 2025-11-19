#!/usr/bin/env python3
"""
Simple ONNX inference server for Go to call.
This is a temporary bridge until full ONNX Runtime Go integration.
"""

import sys
import onnxruntime as ort
import numpy as np
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: onnx_server.py <model_path>", file=sys.stderr)
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Load model
    session = ort.InferenceSession(model_path)
    
    print(f"ONNX Server Ready: {model_path}", file=sys.stderr)
    print("READY", flush=True)
    
    # Read requests from stdin
    for line in sys.stdin:
        try:
            data = json.loads(line)
            
            # Get input array
            input_data = np.array(data['input'], dtype=np.float32)
            
            # Reshape to (1, 1, 80, 16)
            input_tensor = input_data.reshape(1, 1, 80, 16)
            
            # Run inference
            output = session.run(None, {session.get_inputs()[0].name: input_tensor})
            
            # Return result
            result = {
                'output': output[0].flatten().tolist()
            }
            
            print(json.dumps(result), flush=True)
            
        except Exception as e:
            error_result = {'error': str(e)}
            print(json.dumps(error_result), flush=True)

if __name__ == "__main__":
    main()

