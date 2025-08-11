import argparse
import json
import time

import modal

ControlnetsInference = modal.Cls.from_name(
    "controlnets-inference-dev", "ControlnetsInference"
)
infer = ControlnetsInference().run_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Modal ControlnetsInference endpoint"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Number of parallel requests to make",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="test_input.json",
        help="Path to input JSON file",
    )
    return parser.parse_args()


def run_benchmark(num_requests: int, input_file: str):

    # Load test data
    with open(input_file, "r") as f:
        test_data = json.load(f)

    # Track results
    start_time = time.time()

    # Create and spawn tasks
    tasks = []
    for i in range(num_requests):
        # Create a copy of test data and set unique result key for each request
        request_data = test_data.copy()
        request_data["result_key"] = f"test_{i}_{time.time()}"
        print(f"Submitting request {request_data['result_key']}")
        task = infer.spawn(request_data)
        print(f"Task {task.object_id} spawned")
        tasks.append(task)

    # Wait for all tasks to complete
    print("Waiting for all tasks to complete...")
    results = [task.get() for task in tasks]
    print(f"{len(results)} of {num_requests} tasks completed ✓")
    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    avg_time_per_request = total_time / num_requests
    cost_per_second = 0.000542  # L40S
    total_cost = cost_per_second * total_time

    # Print results
    print("\nLocal execution time is highly inaccurate (off by more than 5 seconds!!!)")
    print("\nPlease refer to Modal dashboard for accurate metrics")
    print("\nBenchmark Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Request: {avg_time_per_request:.2f} seconds")
    print(f"Total Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.num_requests, args.input_file)
