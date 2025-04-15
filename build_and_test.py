import numpy as np
import sys
import subprocess

# Programming Assignment 4 tests
# Meant to be run on the Hydra cluster
# You will lose points if your code does not pass these tests

proc = subprocess.Popen(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = proc.communicate()

# Decode the output if needed
stdout = stdout.decode("utf-8")
stderr = stderr.decode("utf-8")
if proc.returncode != 0:
  print("Build failed")
  print(stderr)
  sys.exit()

shared_dir = '/mnt/coe/workspace/ece/ece786-spr24/quamsim/'

test_inputs = ["input_for_qc7_q0_q2_q3_q4_q5_q6.txt",
               "input_for_qc10_q0_q1_q3_q5_q7_q9.txt",
               "input_for_qc12_q6_q7_q8_q9_q10_q11.txt",
               "input_for_qc16_q0_q2_q4_q6_q8_q10.txt"]

expected_outputs = ["output_for_qc7_q0_q2_q3_q4_q5_q6.txt",
                    "output_for_qc10_q0_q1_q3_q5_q7_q9.txt",
                    "output_for_qc12_q6_q7_q8_q9_q10_q11.txt",
                    "output_for_qc16_q0_q2_q4_q6_q8_q10.txt"]

for idx,input in enumerate(test_inputs):
  print(f"Running Test{idx}...")
  proc = subprocess.Popen(["./quamsimV2", shared_dir+test_inputs[idx]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = proc.communicate()

  # Decode the output if needed
  stdout = stdout.decode("utf-8")
  stderr = stderr.decode("utf-8")

  if proc.returncode != 0:
    print(f"Test{idx} failed to run!")
    if stderr:
      print("Error message from subprocess:")
      print(stderr)
    sys.exit()

  stdout_output = stdout
  expected_output = np.genfromtxt(shared_dir+expected_outputs[idx]).reshape(-1,1)
  # Split the stdout into lines
  lines = stdout_output.strip().split('\n')
  # Convert the lines to a 2D list
  cuda_output = np.array([[float(val) for val in line.split()] for line in lines])

  if cuda_output.shape != expected_output.shape:
    print(f"Your shape: {cuda_output.shape}")
    print(f"Expected shape: {expected_output.shape}")
    print("Shapes don't match!!")
    sys.exit()

  diff = np.abs(cuda_output-expected_output)

  if np.all(diff <= 0.002):
    print(f"Test{idx} passed.")
  else:
    print(f"Test{idx} failed.")
    sys.exit()
